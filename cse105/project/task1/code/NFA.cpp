#include "NFA.h"
using namespace std;

NFA::NFA(): states(0), sigma{'a', 'b'}, start(-1), accept(0) {}

void NFA::init() {
  display();
  inputStates();
  inputStart();
  inputAccept();
  inputDelta();
  display();
}

void NFA::inputStates() {
  cout << "Enter the number of states: ";
  cin >> states;
  delta = vector<vector<int>>(states, vector<int>(2, 0));
  display();
}

void NFA::inputStart() {
  cout << "Enter the starting state (0-" << states - 1 << "): ";
  cin >> start;
  display();
}

void NFA::inputAccept() {
  cin.ignore(numeric_limits<streamsize>::max(), '\n');

  cout << "Enter the accepting states (0-" << states - 1 << "): ";
  string line;
  getline(cin, line);
  istringstream iss(line);
  int acceptState;
  while (iss >> acceptState)
    accept |= (1 << acceptState);
  display();
}

void NFA::inputDelta() {
  for (int s = 0; s < states; s++) {
    for (int a = 0; a < sigma.size(); a++) {
      cout << "Enter transition output states (0-" << states - 1 << ")" << endl;
      cout << "State: " << s << ", Input: " << sigma[a] << " -> ";
      string line;
      getline(cin, line);
      istringstream iss(line);
      int outState;
      while (iss >> outState)
        delta[s][a] |= (1 << outState);
      display();
    }
  }
}

string NFA::bin2set(int bin) {
  if (bin == 0) return "{}";
  string str = "{ ";
  int cnt = 0;
  bool first = true;
  while (bin > 0) {
    if (bin & 1) {
      if (!first)
        str += ", ";
      str += to_string(cnt);
      first = false;
    }
    bin >>= 1;
    cnt++;
  }
  str += " }";
  return str;
}

DFA* NFA::convert() {
  return new DFA(states, sigma, delta, start, accept);
}

void NFA::printTable() {
  int width = 15;
  cout << setw(width) << "Q \\ \u03A3";
  for (char a : sigma)
    cout << setw(width) << a;
  cout << "\n\n";

  for (int state = 0; state < states; state++) {
    cout << setw(width) << state;
    for (int sig = 0; sig < sigma.size(); sig++) {
      ostringstream cell;
      cell << bin2set(delta[state][sig]);
      cout << setw(width) << cell.str();
    }
    cout << "\n\n";
  }
}

void NFA::display() {
  system("clear");
  cout << "NFA = (Q, \u03A3, \u03B4, S, F)" << "\n\n";

  cout << "Q: ";
  if (states) {
    cout << "{ ";
    for (int i = 0; i < states - 1; i++) {
      cout << i << ", ";
    }
    cout << states - 1 << " }";
  }
  cout << endl;

  cout << "\u03A3: ";
  if (!sigma.empty()) {
    cout << "{ ";
    for (int i = 0; i < sigma.size() - 1; i++) {
      cout << sigma[i] << ", ";
    }
    cout << sigma[sigma.size() - 1] << " }";
  }
  cout << endl;

  cout << "S: ";
  if (start > -1)
    cout << start;
  cout << endl;

  cout << "F: ";
  if (accept)
    cout << bin2set(accept);
  cout << endl;

  cout << "\u03B4: ";
  if (accept) {
    cout << "\n\n";
    printTable();
  }
  cout << endl;
}
