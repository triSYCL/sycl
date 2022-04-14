//==------------ sycl_mem_obj_t.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __linux__

inline void terminate_xsimk() {}

#else
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>
#include <signal.h>

inline void terminate_xsimk() {
  int my_pid = ::getpid();

  /// tree of processes
  std::unordered_map<int, std::vector<int>> ppid2pids;
  /// set of xsimk processes
  std::unordered_set<int> xsimks;
  std::filesystem::path proc("/proc/");
  /// traverse all processes in /proc building out the tree of processes
  for (const std::filesystem::path &entry :
       std::filesystem::directory_iterator(proc)) {
    std::string s = entry.stem();
    if (s.empty() || !(s.find_first_not_of("0123456789") == std::string::npos))
      continue;
    int pid = std::stoi(s);
    std::ifstream file(entry / "stat");
    if (!file.is_open())
      continue;
    int word_count = 0;
    std::array<std::string, 4> words;
    while (1) {
      file >> words[word_count];
      if (word_count == 3)
        break;
      word_count++;
    }
    if (words[1].empty() || words[3].empty())
      continue;
    if (!(words[3].find_first_not_of("0123456789") == std::string::npos))
      continue;
    int ppid = std::stoi(words[3]);
    if (words[1] == "(xsimk)")
      xsimks.insert(pid);
    assert(pid != ppid);
    ppid2pids[ppid].push_back(pid);
  }

  /// traverse the tree from the current process to terminate xsimks
  auto& root = ppid2pids[my_pid];
  std::vector<int> stack;
  stack.insert(stack.end(), root.begin(), root.end());
  while (!stack.empty()) {
    int pid = stack.back();
    stack.pop_back();
    if (xsimks.count(pid))
      ::kill(pid, 9);
    auto& next = ppid2pids[pid];
    stack.insert(stack.end(), next.begin(), next.end());
  }
}

#endif
