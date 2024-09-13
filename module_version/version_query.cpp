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
  const static int @MODULE_NAME@_VERSION_MAJOR = @MODULE_VERSION_MAJOR@;
  const static int @MODULE_NAME@_VERSION_MINOR = @MODULE_VERSION_MINOR@;
  const static int @MODULE_NAME@_VERSION_PATCH = @MODULE_VERSION_PATCH@;
  const static int @MODULE_NAME@_VERSION_NUMBER = (@MODULE_NAME@_VERSION_MAJOR * 1000000 + @MODULE_NAME@_VERSION_MINOR *  1000 + @MODULE_NAME@_VERSION_PATCH);  
  const static std::string @MODULE_NAME@_VERSION_STRING = "@MODULE_VERSION_STRING@";
  const static std::string @MODULE_NAME@_VERSION_STATUS = "@MODULE_VERSION_STATUS@";
  const static std::string @MODULE_NAME@_GIT_BRANCH = "@GIT_BRANCH@";
  const static std::string @MODULE_NAME@_COMMIT_ID = "@GIT_COMMIT_ID@";
  const static std::string @MODULE_NAME@_COMMIT_DATE = "@GIT_COMMIT_DATE@";
  const static std::string MODULE_NAME = "@MODULE_NAME@";
  inline std::string GetVersionInfo() {
    return std::string("@MODULE_NAME@") + ": " + @MODULE_NAME@_VERSION_STRING + " " + @MODULE_NAME@_VERSION_STATUS
  #ifndef VERSION_WHITOUT_GIT
    + " @ " + @MODULE_NAME@_GIT_BRANCH + " " + @MODULE_NAME@_COMMIT_ID + " at " + @MODULE_NAME@_COMMIT_DATE
  #endif
    ;
  }
} //! namespace @MODULE_NAME@

int main(int argc, char* argv[]) {
  CJsonObject rootNode, newNode;
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
  } else {
    rootNode.Parse("{}");
  }
  
  newNode.Add("version", @MODULE_NAME@::@MODULE_NAME@_VERSION_STRING);
#ifndef VERSION_WHITOUT_GIT
  newNode.Add("git branch", @MODULE_NAME@::@MODULE_NAME@_GIT_BRANCH);
  newNode.Add("commit id", @MODULE_NAME@::@MODULE_NAME@_COMMIT_ID);
  newNode.Add("commit date", @MODULE_NAME@::@MODULE_NAME@_COMMIT_DATE);
#endif
  rootNode.ReplaceAdd(@MODULE_NAME@::MODULE_NAME, newNode);
  std::cout << "@MODULE_NAME@:\n" << newNode.ToFormattedString() << std::endl;
  // std::cout << rootNode.ToFormattedString() << std::endl;

  std::ofstream fout(argv[1]);
  if (fout.good()) {
    fout << rootNode.ToFormattedString();
    fout.close();
  }

  return 0;
}
// clang-format on
