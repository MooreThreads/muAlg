// ##############################################################################
// # @MODULE_NAME@ VERSION INFORMATION
// ##############################################################################
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "Json.hpp"

// clang-format off
namespace @MODULE_NAME@ {
  // const static int @MODULE_NAME@_VERSION_MAJOR = @MODULE_VERSION_MAJOR@;
  // const static int @MODULE_NAME@_VERSION_MINOR = @MODULE_VERSION_MINOR@;
  // const static int @MODULE_NAME@_VERSION_PATCH = @MODULE_VERSION_PATCH@;
  // const static int @MODULE_NAME@_VERSION_NUMBER = (@MODULE_NAME@_VERSION_MAJOR * 1000000 + @MODULE_NAME@_VERSION_MINOR *  1000 + @MODULE_NAME@_VERSION_PATCH);  
  const static std::string @MODULE_NAME@_VERSION_STRING = "@MODULE_VERSION_STRING@";
  const static std::string @MODULE_NAME@_VERSION_STATUS = "@MODULE_VERSION_STATUS@";
  const static std::string @MODULE_NAME@_GIT_BRANCH = "@GIT_BRANCH@";
  const static std::string @MODULE_NAME@_GIT_TAG = "@GIT_TAG@";
  const static std::string @MODULE_NAME@_COMMIT_ID = "@GIT_COMMIT_ID@";
  const static std::string @MODULE_NAME@_COMMIT_DATE = "@GIT_COMMIT_DATE@";
  const static std::string @MODULE_NAME@_BUILD_ARCHS = "@MODULE_BUILD_ARCHS@";
  const static std::string MODULE_NAME = "@MODULE_NAME@";
  const static std::string SHOW_DDK = "@SHOW_DDK@";
  const static std::string DDK_BRANCH = "@DDK_BRANCH@";
  const static std::string DDK_TAG = "@DDK_TAG@";
  const static std::string DDK_COMMIT_ID = "@DDK_COMMIT_ID@";
  const static std::string DDK_COMMIT_DATE = "@DDK_COMMIT_DATE@";
  inline std::string GetVersionInfo() {
    return std::string("@MODULE_NAME@") + ": " + @MODULE_NAME@_VERSION_STRING + " " + @MODULE_NAME@_VERSION_STATUS
  #ifndef VERSION_WHITOUT_GIT
    + " @ " + @MODULE_NAME@_GIT_BRANCH + " " + @MODULE_NAME@_GIT_TAG + " " + @MODULE_NAME@_COMMIT_ID + " at " + @MODULE_NAME@_COMMIT_DATE
  #endif
    ;
  }
} //! namespace @MODULE_NAME@

int main(int argc, char* argv[]) {
  CJsonObject rootNode, newNode, subNode;
  if (argc > 1) {
    std::ifstream fin(argv[1]);
    if (fin.good()) {
      std::stringstream ssContent;
      ssContent << fin.rdbuf();
      std::string input = ssContent.str();
      if (input.empty()) input = "{}";
      if (rootNode.Parse(input)) {
        // std::cout << rootNode.ToFormattedString() << std::endl;
        // std::cout << "==================================" << std::endl;
      } else {
        std::cerr << "parse json error"
                  << "\n";
      }
      fin.close();
    }
  } else {
    rootNode.Parse("{}");
  }
  
  newNode.Add("version", @MODULE_NAME@::@MODULE_NAME@_VERSION_STRING);
  if (std::string(@MODULE_NAME@::@MODULE_NAME@_BUILD_ARCHS).compare("Not Set") != 0) {
    newNode.Add("build archs", @MODULE_NAME@::@MODULE_NAME@_BUILD_ARCHS);
  }
#ifndef VERSION_WHITOUT_GIT
  newNode.Add("git branch", @MODULE_NAME@::@MODULE_NAME@_GIT_BRANCH);
  newNode.Add("git tag", @MODULE_NAME@::@MODULE_NAME@_GIT_TAG);
  newNode.Add("commit id", @MODULE_NAME@::@MODULE_NAME@_COMMIT_ID);
  newNode.Add("commit date", @MODULE_NAME@::@MODULE_NAME@_COMMIT_DATE);
#endif
  rootNode.ReplaceAdd(@MODULE_NAME@::MODULE_NAME, newNode);
  std::cout << "@MODULE_NAME@:\n" << newNode.ToFormattedString() << std::endl;

#ifndef VERSION_WHITOUT_GIT
  if (std::string(@MODULE_NAME@::SHOW_DDK).compare("ON") == 0) {
    subNode.Add("git branch", @MODULE_NAME@::DDK_BRANCH);
    subNode.Add("git tag", @MODULE_NAME@::DDK_TAG);
    subNode.Add("commit id", @MODULE_NAME@::DDK_COMMIT_ID);
    subNode.Add("commit date", @MODULE_NAME@::DDK_COMMIT_DATE);
    std::cout << "driver_dependency:\n" << subNode.ToFormattedString() << std::endl;
  }
#endif
  if (argc > 1) {
    std::ofstream fout(argv[1]);
    if (fout.good()) {
      fout << rootNode.ToFormattedString();
      fout.close();
    }
  }

  return 0;
}
// clang-format on
