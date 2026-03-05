#!/usr/bin/env python3
"""
CUB 测试统计脚本 v3 - 每GPU独立队列，失败后分发到其他GPU重试

功能：
1. 清理并重新配置 CMake（使用_no_thrust版本）
2. 编译所有测试
3. 每个GPU维护独立任务队列，队列内串行执行
4. 测试失败后分发到其他2个GPU重试
5. 统计通过率，生成详细报告

使用示例:
  # 运行所有测试（8 GPU并行，每GPU队列内串行）
  python3 stat_cub_tests_v3.py

  # 只运行 device_scan 相关的测试程序
  python3 stat_cub_tests_v3.py -k scan

  # 跳过清理和编译，直接运行测试（用于调试）
  python3 stat_cub_tests_v3.py --skip-clean --skip-configure --skip-compile -k scan
"""

import os
import sys
import subprocess
import shutil
import time
import threading
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random


class TestResult(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    EXCEPTION = "EXCEPTION"


@dataclass
class FailedCase:
    """单个失败的测试用例"""
    test_info: str
    error_detail: str


@dataclass
class TestAttempt:
    """单次测试尝试的结果"""
    gpu_id: int
    result: TestResult
    elapsed_time: float
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    failed_case_details: List[FailedCase] = field(default_factory=list)
    error_message: str = ""
    output: str = ""


@dataclass
class TestInfo:
    """测试的完整信息（包含所有尝试）"""
    name: str
    path: str
    attempts: List[TestAttempt] = field(default_factory=list)
    
    @property
    def final_result(self) -> TestResult:
        """最终结果：任意一次通过即为通过"""
        for a in self.attempts:
            if a.result == TestResult.PASSED:
                return TestResult.PASSED
        # 都没通过，返回最后一次的结果
        return self.attempts[-1].result if self.attempts else TestResult.FAILED
    
    @property
    def best_attempt(self) -> Optional[TestAttempt]:
        """最佳尝试（通过的优先，否则取最后一次）"""
        for a in self.attempts:
            if a.result == TestResult.PASSED:
                return a
        return self.attempts[-1] if self.attempts else None
    
    @property
    def total_cases(self) -> int:
        return self.best_attempt.total_cases if self.best_attempt else 0
    
    @property
    def passed_cases(self) -> int:
        return self.best_attempt.passed_cases if self.best_attempt else 0
    
    @property
    def failed_cases(self) -> int:
        return self.best_attempt.failed_cases if self.best_attempt else 0
    
    @property
    def failed_case_details(self) -> List[FailedCase]:
        return self.best_attempt.failed_case_details if self.best_attempt else []
    
    @property
    def retry_count(self) -> int:
        return len(self.attempts) - 1
    
    @property
    def tried_gpus(self) -> List[int]:
        return [a.gpu_id for a in self.attempts]


class GpuTaskQueue:
    """单个GPU的任务队列"""
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.queue: deque = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.finished = False  # 标记是否还有新任务
    
    def put(self, task_name: str):
        """添加任务到队列"""
        with self.lock:
            self.queue.append(task_name)
            self.condition.notify()
    
    def get(self, timeout: float = 1.0) -> Optional[str]:
        """从队列获取任务，队列空且finished时返回None"""
        with self.lock:
            while not self.queue:
                if self.finished:
                    return None
                if not self.condition.wait(timeout=timeout):
                    # 超时后再次检查
                    if self.finished:
                        return None
                    continue
            if self.queue:
                return self.queue.popleft()
            return None
    
    def size(self) -> int:
        with self.lock:
            return len(self.queue)
    
    def close(self):
        """标记队列不再接收新任务"""
        with self.lock:
            self.finished = True
            self.condition.notify_all()


class TaskScheduler:
    """任务调度器：管理多GPU任务队列和重试逻辑"""
    def __init__(self, num_gpus: int, max_retries: int = 2):
        self.num_gpus = num_gpus
        self.max_retries = max_retries  # 额外重试次数（不含首次）
        self.queues: List[GpuTaskQueue] = [GpuTaskQueue(i) for i in range(num_gpus)]
        self.all_tasks: List[str] = []
        
        # 结果收集
        self.results: Dict[str, TestInfo] = {}
        self.results_lock = threading.Lock()
        
        # 任务状态追踪
        self.task_status: Dict[str, str] = {}  # task_name -> "pending" | "running" | "done"
        self.task_attempts: Dict[str, Set[int]] = {}  # task_name -> set of tried gpu_ids
        self.task_lock = threading.Lock()
        
        # 统计
        self.completed_count = 0
        self.completed_lock = threading.Lock()
    
    def distribute_tasks(self, tasks: List[str]):
        """初始分配任务到各GPU队列（相对平均）"""
        self.all_tasks = tasks
        random.shuffle(tasks)  # 打乱以分散负载
        
        for i, task in enumerate(tasks):
            gpu_id = i % self.num_gpus
            self.queues[gpu_id].put(task)
            with self.task_lock:
                self.task_status[task] = "pending"
                self.task_attempts[task] = set()
        
        print(f"[调度器] 已将 {len(tasks)} 个任务分配到 {self.num_gpus} 个GPU队列")
        for i, q in enumerate(self.queues):
            print(f"  GPU {i}: {q.size()} 个任务")
    
    def schedule_retry(self, task_name: str, failed_gpu: int):
        """将失败的任务调度到其他GPU重试"""
        with self.task_lock:
            attempts = self.task_attempts.get(task_name, set())
            attempts.add(failed_gpu)
            
            # 检查是否已达到最大重试次数
            if len(attempts) >= self.max_retries + 1:
                return False
            
            # 选择2个未尝试过的GPU
            available_gpus = [i for i in range(self.num_gpus) if i not in attempts]
            if not available_gpus:
                return False
            
            # 选择前2个可用GPU（如果有的话）
            retry_gpus = available_gpus[:min(2, len(available_gpus))]
            
            for gpu_id in retry_gpus:
                self.queues[gpu_id].put(task_name)
                self.task_status[task_name] = "retry"
                self.task_attempts[task_name].add(gpu_id)
            
            return True
    
    def mark_task_done(self, task_name: str, test_info: TestInfo) -> bool:
        """标记任务完成，返回True表示是首次完成，False表示已被其他GPU完成"""
        with self.task_lock:
            if self.task_status.get(task_name) == "done":
                # 已经被其他GPU标记完成，只合并尝试记录
                with self.results_lock:
                    if task_name in self.results:
                        self.results[task_name].attempts.extend(test_info.attempts)
                return False
            self.task_status[task_name] = "done"
        
        with self.results_lock:
            if task_name not in self.results:
                self.results[task_name] = test_info
            else:
                self.results[task_name].attempts.extend(test_info.attempts)
        
        with self.completed_lock:
            self.completed_count += 1
        return True
    
    def get_pending_attempts(self, task_name: str) -> Set[int]:
        """获取任务已尝试过的GPU集合"""
        with self.task_lock:
            return self.task_attempts.get(task_name, set()).copy()
    
    def close_all(self):
        """关闭所有队列"""
        for q in self.queues:
            q.close()
    
    def progress(self) -> Tuple[int, int]:
        """返回 (已完成, 总数)"""
        with self.completed_lock:
            return self.completed_count, len(self.all_tasks)


class TestRunner:
    def __init__(self, cub_dir: str, num_gpus: int = 8, timeout: int = 120,
                 keyword: str = None, case_keyword: str = None):
        self.cub_dir = Path(cub_dir).resolve()
        self.build_dir = self.cub_dir / "build"
        self.bin_dir = self.build_dir / "bin"
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.keyword = keyword.lower() if keyword else None
        self.case_keyword = case_keyword.lower() if case_keyword else None
        self.log_file = Path(__file__).parent / "test_stats.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_lock = threading.Lock()
    
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        with self.log_lock:
            print(line)
            with open(self.log_file, "a") as f:
                f.write(line + "\n")
    
    def clean(self):
        self.log("=" * 60)
        self.log("步骤1: 清理build目录")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            self.log("已删除build目录")
    
    def configure(self) -> bool:
        self.log("=" * 60)
        self.log("步骤2: CMake配置")
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = ["cmake", "-DMUSA_64_BIT_DEVICE_CODE=ON", "-DCUB_ENABLE_TESTING=ON", str(self.cub_dir)]
        self.log(f"执行: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=self.build_dir, capture_output=True, text=True, timeout=300)
        
        skipped = [l for l in result.stdout.split("\n") if "Skipping" in l and "no-thrust" in l]
        self.log(f"跳过 {len(skipped)} 个使用thrust的测试")
        
        cache_file = self.build_dir / "CMakeCache.txt"
        
        if result.returncode != 0 or not cache_file.exists():
            self.log(f"CMake配置失败")
            self.log(f"stderr: {result.stderr[:1000]}")
            return False
        
        self.log("CMake配置成功")
        return True
    
    def compile(self) -> bool:
        self.log("=" * 60)
        self.log("步骤3: 编译")
        cmd = ["cmake", "--build", str(self.build_dir), "-j", "1"]
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        
        current_progress = 0
        current_target = ""
        error_output = []
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.rstrip()
                progress_match = re.search(r'\[\s*(\d+)%\]', line)
                if progress_match:
                    current_progress = int(progress_match.group(1))
                    if "Building" in line or "Linking" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p in ["Building", "Linking"]:
                                target_parts = parts[i+3:] if i+3 < len(parts) else []
                                current_target = " ".join(target_parts[:3])[:50]
                                break
                    bar_len = 30
                    filled = int(bar_len * current_progress / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    target_display = current_target if current_target else "准备中..."
                    print(f"\r[{bar}] {current_progress:3d}% | {target_display}", end="", flush=True)
                elif "error:" in line.lower():
                    error_output.append(line)
        
        print()
        _, stderr = process.communicate()
        if stderr:
            error_output.append(stderr)
        
        if process.returncode != 0:
            self.log(f"编译有错误，继续运行已编译的测试...")
        
        binaries = list(self.bin_dir.glob("*")) if self.bin_dir.exists() else []
        self.log(f"编译完成，共 {len(binaries)} 个二进制文件")
        return True
    
    def discover_tests(self) -> List[str]:
        if not self.bin_dir.exists():
            return []
        tests = []
        for f in self.bin_dir.glob("cub.cpp17.test.*"):
            if f.is_file() and os.access(f, os.X_OK):
                if self.keyword and self.keyword not in f.name.lower():
                    continue
                tests.append(f.name)
        return sorted(tests)
    
    def parse_test_output(self, output: str) -> Tuple[int, int, int, List[FailedCase]]:
        """解析测试输出，支持多种PASS/FAIL格式"""
        lines = output.split('\n')
        pass_count = 0
        fail_count = 0
        failed_cases = []
        current_test_info = ""
        current_matches = True
        
        for line in lines:
            # 提取测试描述行
            if any(kw in line for kw in ["Pointer", "Iterator", "CUB_"]) and ("cub::" in line or "Device" in line):
                current_test_info = line.strip()
                if self.case_keyword:
                    current_matches = self.case_keyword in current_test_info.lower()
                else:
                    current_matches = True
            
            if not current_matches:
                continue
            
            # 统计 PASS - 支持多种格式：
            # - "\tPASS" (tab + PASS)
            # - "PASS" at line end
            # - "PASS " with trailing space (radix_sort format)
            # - ": PASS " or ": PASS\n"
            if "PASS" in line:
                stripped = line.strip()
                if ("\tPASS" in line or 
                    stripped.endswith("PASS") or 
                    ": PASS " in line or
                    re.search(r':\s*PASS\s*$', line)):
                    pass_count += 1
            
            # 检测失败
            if "INCORRECT:" in line or "FAIL" in line or "AssertEquals" in line:
                fail_count += 1
                error_match = re.search(r'INCORRECT:\s*(.+)', line)
                if error_match:
                    error_detail = error_match.group(1).strip()
                elif "AssertEquals" in line:
                    error_detail = line.strip()
                else:
                    error_detail = line.strip()
                
                if current_test_info:
                    failed_cases.append(FailedCase(
                        test_info=current_test_info,
                        error_detail=error_detail
                    ))
        
        total = pass_count + fail_count
        return total, pass_count, fail_count, failed_cases
    
    def run_single_test(self, test_name: str, gpu_id: int) -> TestAttempt:
        """在指定GPU上运行单个测试"""
        test_path = self.bin_dir / test_name
        env = os.environ.copy()
        env["MUSA_VISIBLE_DEVICES"] = str(gpu_id)
        
        attempt = TestAttempt(gpu_id=gpu_id, result=TestResult.FAILED, elapsed_time=0)
        
        try:
            start = time.time()
            result = subprocess.run(
                [str(test_path)],
                capture_output=True, text=True, timeout=self.timeout, env=env
            )
            attempt.elapsed_time = time.time() - start
            attempt.output = result.stdout + result.stderr
            
            total, passed, failed, failed_cases = self.parse_test_output(attempt.output)
            attempt.total_cases = total
            attempt.passed_cases = passed
            attempt.failed_cases = failed
            attempt.failed_case_details = failed_cases
            
            if failed > 0 or result.returncode != 0:
                attempt.result = TestResult.FAILED
                attempt.error_message = f"{failed}/{total} cases failed"
            elif passed > 0:
                attempt.result = TestResult.PASSED
            else:
                attempt.result = TestResult.PASSED  # 无case输出的视为通过
        
        except subprocess.TimeoutExpired:
            attempt.elapsed_time = self.timeout
            attempt.result = TestResult.TIMEOUT
            attempt.error_message = f"超时({self.timeout}秒)"
        except Exception as e:
            attempt.elapsed_time = 0
            attempt.result = TestResult.EXCEPTION
            attempt.error_message = str(e)
        
        return attempt
    
    def gpu_worker(self, scheduler: TaskScheduler, gpu_id: int):
        """GPU工作线程：从自己的队列取任务执行"""
        queue = scheduler.queues[gpu_id]
        
        while True:
            task_name = queue.get(timeout=2.0)
            if task_name is None:
                # 队列为空且已关闭
                break
            
            # 检查是否已经完成（可能是重试时其他GPU已完成）
            with scheduler.task_lock:
                if scheduler.task_status.get(task_name) == "done":
                    continue
            
            # 记录本次尝试
            with scheduler.task_lock:
                if task_name not in scheduler.task_attempts:
                    scheduler.task_attempts[task_name] = set()
                scheduler.task_attempts[task_name].add(gpu_id)
            
            # 运行测试
            attempt = self.run_single_test(task_name, gpu_id)
            
            # 构建TestInfo
            test_path = self.bin_dir / task_name
            test_info = TestInfo(name=task_name, path=str(test_path))
            test_info.attempts.append(attempt)
            
            # 判断是否需要重试
            need_retry = False
            if attempt.result != TestResult.PASSED:
                # 尝试调度到其他GPU重试
                need_retry = scheduler.schedule_retry(task_name, gpu_id)
            
            # 如果不需要重试或重试调度失败，标记完成
            if not need_retry:
                is_first = scheduler.mark_task_done(task_name, test_info)
            else:
                is_first = False  # 重试任务，本次不计入completed
            
            # 输出进度（只有首次完成才计入进度）
            if is_first or attempt.result == TestResult.PASSED:
                completed, total = scheduler.progress()
                status = "✓" if attempt.result == TestResult.PASSED else "✗"
                case_info = f" ({attempt.passed_cases}/{attempt.total_cases})" if attempt.total_cases > 0 else ""
                retry_mark = " [重试]" if not is_first and attempt.result == TestResult.PASSED else ""
                self.log(f"[{completed}/{total}] GPU{gpu_id} {status} {task_name}{case_info}{retry_mark} ({attempt.elapsed_time:.1f}s)")
    
    def run_all_tests(self, output_log=None, max_retries=2):
        """运行所有测试"""
        self.log("=" * 60)
        self.log("步骤4: 运行测试")
        
        tests = self.discover_tests()
        if not tests:
            self.log("没有发现测试")
            return
        
        self.log(f"发现 {len(tests)} 个测试")
        self.log(f"重试策略: 失败后最多分发到 {max_retries} 个其他GPU")
        
        # 创建调度器
        scheduler = TaskScheduler(num_gpus=self.num_gpus, max_retries=max_retries)
        scheduler.distribute_tasks(tests)
        
        # 启动GPU工作线程
        threads = []
        for gpu_id in range(self.num_gpus):
            t = threading.Thread(target=self.gpu_worker, args=(scheduler, gpu_id))
            t.start()
            threads.append(t)
        
        # 等待所有任务完成，然后关闭队列让线程退出
        while True:
            completed, total = scheduler.progress()
            if completed >= total:
                break
            time.sleep(1.0)
        
        # 关闭所有队列，让工作线程退出
        scheduler.close_all()
        
        # 等待所有线程完成（带超时保护）
        for t in threads:
            t.join(timeout=10.0)
            if t.is_alive():
                self.log(f"警告: 线程未能正常退出，强制继续")
        
        # 收集结果
        self.results = list(scheduler.results.values())
        
        # 保存输出日志
        if output_log:
            all_outputs = []
            for test_info in sorted(self.results, key=lambda x: x.name):
                all_outputs.append(f"\n{'='*60}\n")
                all_outputs.append(f"Test: {test_info.name}\n")
                all_outputs.append(f"Result: {'✓' if test_info.final_result == TestResult.PASSED else '✗'}\n")
                all_outputs.append(f"Attempts: {len(test_info.attempts)} (GPUs: {test_info.tried_gpus})\n")
                all_outputs.append(f"{'='*60}\n")
                if test_info.best_attempt:
                    all_outputs.append(test_info.best_attempt.output)
            
            with open(output_log, 'w', encoding='utf-8') as f:
                f.write(''.join(all_outputs))
            self.log(f"输出日志已保存: {output_log}")
    
    def generate_report(self) -> str:
        self.log("=" * 60)
        self.log("步骤5: 生成报告")
        
        total_programs = len(self.results)
        passed_programs = sum(1 for r in self.results if r.final_result == TestResult.PASSED)
        failed_programs = sum(1 for r in self.results if r.final_result == TestResult.FAILED)
        timeout_programs = sum(1 for r in self.results if r.final_result == TestResult.TIMEOUT)
        exception_programs = sum(1 for r in self.results if r.final_result == TestResult.EXCEPTION)
        
        total_cases = sum(r.total_cases for r in self.results)
        total_passed_cases = sum(r.passed_cases for r in self.results)
        total_failed_cases = sum(r.failed_cases for r in self.results)
        
        program_pass_rate = (passed_programs / total_programs * 100) if total_programs > 0 else 0
        case_pass_rate = (total_passed_cases / total_cases * 100) if total_cases > 0 else 0
        
        report = []
        report.append("# CUB 测试统计报告 v3 (多GPU队列模式)")
        report.append("")
        report.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        filters = []
        if self.keyword:
            filters.append(f"**程序过滤:** `{self.keyword}`")
        if self.case_keyword:
            filters.append(f"**用例过滤:** `{self.case_keyword}`")
        if filters:
            report.append(" | ".join(filters))
            report.append("")
        
        report.append("---")
        report.append("")
        
        # 表格1: 测试程序统计
        report.append("## 表格1: 测试程序统计")
        report.append("")
        
        test_categories = {}
        for r in self.results:
            parts = r.name.replace("cub.cpp17.test.", "").split(".")
            category = parts[0] if parts else "unknown"
            if category not in test_categories:
                test_categories[category] = {"passed": 0, "failed": 0, "timeout": 0, "exception": 0, "total": 0}
            test_categories[category]["total"] += 1
            if r.final_result == TestResult.PASSED:
                test_categories[category]["passed"] += 1
            elif r.final_result == TestResult.TIMEOUT:
                test_categories[category]["timeout"] += 1
            elif r.final_result == TestResult.EXCEPTION:
                test_categories[category]["exception"] += 1
            else:
                test_categories[category]["failed"] += 1
        
        report.append("| 测试类型 | 总数 | 通过 | 失败 | 超时 | 异常 | 通过率 |")
        report.append("|:---------|-----:|-----:|-----:|-----:|-----:|-------:|")
        
        for category in sorted(test_categories.keys()):
            stats = test_categories[category]
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report.append(f"| {category} | {stats['total']} | {stats['passed']} | {stats['failed']} | {stats['timeout']} | {stats['exception']} | {rate:.1f}% |")
        
        report.append(f"| **总计** | **{total_programs}** | **{passed_programs}** | **{failed_programs}** | **{timeout_programs}** | **{exception_programs}** | **{program_pass_rate:.1f}%** |")
        report.append("")
        
        # 表格2: 详细测试用例统计
        report.append("## 表格2: 测试用例详细统计")
        report.append("")
        report.append("| 状态 | 测试程序 | 总Case | 通过 | 失败 | 通过率 | 尝试GPU |")
        report.append("|:----:|:---------|-------:|-----:|-----:|-------:|:--------|")
        
        for r in sorted(self.results, key=lambda x: x.name):
            short_name = r.name.replace("cub.cpp17.test.", "")
            status = "✓" if r.final_result == TestResult.PASSED else "✗"
            gpu_info = ",".join(map(str, r.tried_gpus))
            if r.total_cases > 0:
                rate = (r.passed_cases / r.total_cases * 100) if r.total_cases > 0 else 0
                report.append(f"| {status} | {short_name} | {r.total_cases} | {r.passed_cases} | {r.failed_cases} | {rate:.1f}% | {gpu_info} |")
            else:
                report.append(f"| {status} | {short_name} | N/A | N/A | N/A | N/A | {gpu_info} |")
        
        report.append(f"| | **总计** | **{total_cases}** | **{total_passed_cases}** | **{total_failed_cases}** | **{case_pass_rate:.1f}%** | |")
        report.append("")
        
        # 表格3: 失败用例详情
        failed_tests = [r for r in self.results if r.failed_cases > 0 or r.final_result in [TestResult.TIMEOUT, TestResult.EXCEPTION]]
        
        if failed_tests:
            report.append("## 表格3: 失败用例详情")
            report.append("")
            
            for r in sorted(failed_tests, key=lambda x: x.name):
                short_name = r.name.replace("cub.cpp17.test.", "")
                report.append(f"### {short_name}")
                report.append("")
                
                if r.final_result == TestResult.TIMEOUT:
                    report.append(f"> **TIMEOUT:** {r.best_attempt.error_message if r.best_attempt else '超时'}")
                elif r.final_result == TestResult.EXCEPTION:
                    report.append(f"> **EXCEPTION:** {r.best_attempt.error_message if r.best_attempt else '异常'}")
                else:
                    report.append(f"**失败:** {r.failed_cases}/{r.total_cases} cases (尝试GPU: {r.tried_gpus})")
                    report.append("")
                    
                    if r.failed_case_details:
                        # 只显示前10个，避免报告过长
                        shown_cases = r.failed_case_details[:10]
                        report.append("| 测试描述 | 错误信息 |")
                        report.append("|:---------|:---------|")
                        for fc in shown_cases:
                            error_escaped = fc.error_detail.replace("|", "\\|")
                            report.append(f"| {fc.test_info} | `{error_escaped}` |")
                        if len(r.failed_case_details) > 10:
                            report.append(f"| ... | (还有 {len(r.failed_case_details) - 10} 个失败用例) |")
                
                report.append("")
        
        report.append("---")
        report.append("")
        report.append("## 总结")
        report.append("")
        report.append(f"- **测试程序:** {passed_programs}/{total_programs} 通过 ({program_pass_rate:.1f}%)")
        report.append(f"- **测试用例:** {total_passed_cases}/{total_cases} 通过 ({case_pass_rate:.1f}%)")
        if failed_programs > 0:
            report.append(f"- 有 **{failed_programs}** 个测试程序存在失败用例")
        report.append("")
        
        report_str = "\n".join(report)
        
        report_file = Path(__file__).parent / "test_report.md"
        with open(report_file, "w") as f:
            f.write(report_str)
        self.log(f"报告已保存: {report_file}")
        
        return report_str


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CUB测试统计脚本v3 - 每GPU独立队列")
    parser.add_argument("--skip-clean", action="store_true", help="跳过清理步骤")
    parser.add_argument("--skip-configure", action="store_true", help="跳过CMake配置")
    parser.add_argument("--skip-compile", action="store_true", help="跳过编译步骤")
    parser.add_argument("-k", "--keyword", type=str, default=None,
                        help="测试程序关键字过滤")
    parser.add_argument("-c", "--case-keyword", type=str, default=None,
                        help="测试用例关键字过滤")
    parser.add_argument("-r", "--max-retries", type=int, default=2,
                        help="失败后额外重试次数 (默认: 2, 即最多在3个GPU上尝试)")
    parser.add_argument("-n", "--num-gpus", type=int, default=8,
                        help="GPU数量 (默认: 8)")
    parser.add_argument("-t", "--timeout", type=int, default=1500,
                        help="单个测试超时时间(秒) (默认: 1500，与ctest一致)")
    parser.add_argument("-o", "--output-log", type=str, default=None,
                        help="保存完整输出日志到指定文件")
    args = parser.parse_args()
    
    runner = TestRunner(
        cub_dir=".",
        num_gpus=args.num_gpus,
        timeout=args.timeout,
        keyword=args.keyword,
        case_keyword=args.case_keyword
    )
    
    if args.keyword:
        runner.log(f"测试程序过滤: {args.keyword}")
    if args.case_keyword:
        runner.log(f"测试用例过滤: {args.case_keyword}")
    if args.output_log:
        runner.log(f"输出日志: {args.output_log}")
    
    if not args.skip_clean:
        runner.clean()
    else:
        runner.log("跳过清理")
    
    if not args.skip_configure:
        if not runner.configure():
            return
    else:
        runner.log("跳过配置")
    
    if not args.skip_compile:
        runner.compile()
    else:
        runner.log("跳过编译")
    
    start = time.time()
    runner.run_all_tests(output_log=args.output_log, max_retries=args.max_retries)
    report = runner.generate_report()
    
    print("\n" + report)
    
    runner.log("=" * 60)
    passed = sum(1 for r in runner.results if r.final_result == TestResult.PASSED)
    total = len(runner.results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    runner.log(f"完成! 程序通过率: {pass_rate:.2f}% ({passed}/{total})")
    runner.log(f"总耗时: {time.time() - start:.1f}秒")


if __name__ == "__main__":
    main()
