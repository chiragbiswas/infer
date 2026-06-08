CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native \
            -Wall -Wextra -Iinclude
LDFLAGS  :=

TARGETS  := infer lm attn
SRC_INFER := src/main.cpp
SRC_LM    := src/lm.cpp
HEADERS   := include/tensor.h include/ops.h include/net.h include/embedding.h include/charset.h include/attention.h

.PHONY: all clean

all: $(TARGETS)

infer: $(SRC_INFER) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_INFER) $(LDFLAGS)

lm: $(SRC_LM) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_LM) $(LDFLAGS)

attn: src/attn.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ src/attn.cpp $(LDFLAGS)

clean:
	rm -f $(TARGETS)
