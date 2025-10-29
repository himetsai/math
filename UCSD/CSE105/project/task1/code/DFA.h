#ifndef DFA_H
#define DFA_H

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;

class DFA {
    
	int states;
	vector<char> sigma;
	vector<vector<int>> delta;
	int start;
	int accept;

	public:

		DFA(int NFAstates, vector<char>& NFAsigma, vector<vector<int>>& NFAdelta, int NFAstart, int NFAaccept);

		string bin2set(int bin);

		void printTable();

		void display();
};

#endif