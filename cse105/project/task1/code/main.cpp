#include <iostream>
#include "NFA.h"
#include "DFA.h"
using namespace std;

int main() {
	NFA nfa = NFA();

	nfa.init();

	cout << endl;

	DFA* dfa = nfa.convert();
	dfa->display();

	return 0;
}