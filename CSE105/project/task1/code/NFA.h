#ifndef NFA_H
#define NFA_H

#include "DFA.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
using namespace std;

class NFA {

	int states;
	vector<char> sigma;
	vector<vector<int>> delta;
	int start;
	int accept;

	public:

	NFA();

	void init();

	void inputStates();

	void inputStart();

	void inputAccept();

	void inputDelta();

	string bin2set(int bin);

	void printTable();

	void display();

	DFA* convert();
};

#endif