#include "io/mmf.hpp"

namespace util {
namespace io {

static std::vector<std::string> split(const std::string &str,
                                      const std::string &delim) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;

  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos)
      pos = str.length();
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty())
      tokens.push_back(token);
    prev = pos + delim.length();
  }

  while (pos < str.length() && prev < str.length());
  return tokens;
}

// Returns false at EOF
bool DoRead(std::ifstream &in, std::vector<std::string> &arguments) {
  std::string buff;

  try {
    if (getline(in, buff).eof()) {
      return false;
    }
  } catch (std::ios_base::failure &e) {
    // cout << "error reading from MMF file:" + (std::string) e.what() + "\n";
    exit(1);
  }

  buff.erase(0, buff.find_first_not_of(" \t"));
  buff.erase(buff.find_last_not_of(" \t") + 1,
             buff.size() - buff.find_last_not_of(" \t") - 1);
  arguments = split(buff, " ");

  return true;
}

// Explicit template instantiations
template struct MMF<int, float>;
template struct MMF<int, double>;

} // end of namespace io
} // end of namespace util
