GAFERMION=gafermion

CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC
CXXFLAGS += $(shell pkg-config --cflags eigen3)

PYTHON_CFLAGS += $(shell python3-config --cflags)
PYTHON_CFLAGS += $(shell python3 -m pybind11 --includes)

LIB = $(GAFERMION)$(shell python3 -m pybind11 --extension-suffix)

fermion:
	$(CXX) $(CXXFLAGS) $(PYTHON_CFLAGS) pymain.cpp gafermion.cpp -o $(LIB)

clean:
	rm -rf $(LIB)
