#include "DFA.h"
using namespace std;

DFA::DFA(int NFAstates, vector<char>& NFAsigma, vector<vector<int>>& NFAdelta, int NFAstart, int NFAaccept): start(NFAstart), accept(NFAaccept) {
  states = 1 << NFAstates;
  sigma = NFAsigma;
  delta = vector<vector<int>>(states, vector<int>(2, 0));
  for (int s = 0; s < states; s++) {
    for (int a = 0; a < sigma.size(); a++) {
      int temp = s;
      int cnt = 0;
      while (temp > 0) {
        if (temp & 1)
          delta[s][a] |= NFAdelta[cnt][a];
        cnt++;
        temp >>= 1;
      }
    }
  }
};

string DFA::bin2set(int bin) {
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

void DFA::printTable() {
  int width = 15;
  cout << setw(width) << "Q \\ \u03A3";
  for (char a : sigma) {
    cout << setw(width) << a;
  }
  cout << "\n\n";

  for (int s = 0; s < states; s++) {
    ostringstream cell;
    cell << bin2set(s);
    cout << setw(width) << cell.str();

    for (int a = 0; a < sigma.size(); a++) {
      ostringstream cell;
      cell << bin2set(delta[s][a]);
      cout << setw(width) << cell.str();
    }
    cout << "\n\n";
  }
};

void DFA::display() {
  cout << "DFA = (Q, \u03A3, \u03B4, S, F)" << "\n\n";

  cout << "Q: { ";
  for (int i = 0; i < states - 1; i++)
    cout << bin2set(i) << ", ";
  cout << bin2set(states - 1) << " }" << endl;

  cout << "\u03A3: { ";
  for (int i = 0; i < sigma.size() - 1; i++)
    cout << sigma[i] << ", ";
  cout << sigma[sigma.size() - 1] << " }" << endl;

  cout << "S: { " << start << " }" << endl;

  cout << "F: { ";
  bool first = false;
  for (int i = 0; i < states; i++) {
    if (i & accept) {
      if (first)
        cout << ", ";
      first = true;
      cout << bin2set(i);
    }
  }
  cout << " }" << endl;

  cout << "\u03B4: ";
  cout << "\n\n";
  printTable();
  cout << endl;
};