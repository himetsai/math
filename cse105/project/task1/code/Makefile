CXX = clang++
CXXFLAGS = -Wall -pedantic -g -O0 -std=c++11
TARGET = convert
OBJ = main.o NFA.o DFA.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

main.o: main.cpp NFA.h DFA.h
	$(CXX) $(CXXFLAGS) -c main.cpp

NFA.o: NFA.cpp NFA.h
	$(CXX) $(CXXFLAGS) -c NFA.cpp

DFA.o: DFA.cpp DFA.h
	$(CXX) $(CXXFLAGS) -c DFA.cpp

clean:
	rm -f $(TARGET) $(OBJ)
