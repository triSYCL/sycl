// If you are having issues with xsim processed staying alive.
// you can compile this and keep it running in the background.

// clang++ -std=c++17 cleanup_xsimk.cpp -o kill_xsim
// maybe add a -lstdc++-fs if using libstdc++ 7 or 8

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
#include <chrono>
#include <thread>
#include <iostream>

using namespace std::chrono_literals;

inline void terminate_xsimk() {
  /// Tree of processes according to their parenthood
  std::unordered_map<pid_t, pid_t> pid2ppid;
  /// set of xsimk processes
  std::vector<pid_t> xsimks;
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
        xsimks.push_back(pid);
      pid2ppid[pid] = parent_pid;
    } catch (...) {
      continue;
    }
  }

  for (auto pid : xsimks) {
    int count_level = 0;
    pid_t curr = pid;
    // Only kill xsim that are not in use. We consider an xsim in use if on of
    // the 6 parent process of xsim is init(pid == 1)
    for (;curr != 1 && count_level < 7; count_level++)
      curr = pid2ppid[curr];
    if (count_level < 7)
      ::kill(pid, SIGKILL);
  }
}

int main() {
  std::cout << "this process should be launched in background" << std::endl;
  while (1) {
    terminate_xsimk();
    std::this_thread::sleep_for(5min);
  }
}
