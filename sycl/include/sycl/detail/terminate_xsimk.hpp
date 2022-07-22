//==------------ terminate_xsimk.hpp - cleanup HDL simulator ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

/// The implementation of terminate_xsimk depends on many Linux or POSIX
/// specific features so it is only provided on Linux
#ifndef __linux__

inline void terminate_xsimk() {}

#else
#include <cassert>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

inline void terminate_xsimk() {
  pid_t my_pid = ::getpid();

  /// Tree of processes according to their parenthood
  std::unordered_map<pid_t, std::vector<pid_t>> ppid2pids;
  /// set of xsimk processes
  std::unordered_set<pid_t> xsimks;
  std::filesystem::path proc("/proc");
  /// traverse all processes in /proc building out the tree of processes
  for (const std::filesystem::path &entry :
       std::filesystem::directory_iterator(proc)) {
    std::string s = entry.stem();
    try {
      pid_t pid = std::stoi(s);
      std::ifstream file(entry / "stat");
      file.exceptions(std::ifstream::failbit);
      file >> pid;
      std::string short_command_name;
      file >> short_command_name;
      std::string stat;
      file >> stat;
      pid_t parent_pid;
      file >> parent_pid;
      if (short_command_name == "(xsimk)")
        xsimks.insert(pid);
      ppid2pids[parent_pid].push_back(pid);
    } catch (...) {
      continue;
    }
  }

  /// traverse the tree from the current process to terminate our xsimks
  auto &root = ppid2pids[my_pid];
  std::vector<int> stack;
  stack.insert(stack.end(), root.begin(), root.end());
  while (!stack.empty()) {
    int pid = stack.back();
    stack.pop_back();
    if (xsimks.count(pid))
      ::kill(pid, SIGKILL);
    auto &next = ppid2pids[pid];
    stack.insert(stack.end(), next.begin(), next.end());
  }
}

#endif
