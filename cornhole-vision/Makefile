CC      := gcc
CFLAGS  := -O2 -Wall -Wextra -std=c17
BIN     := build/cornhole_vision
SRC     := src/main.c

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p build
	$(CC) $(CFLAGS) -o $(BIN) $(SRC)

run: all
	./$(BIN) /tmp/cornhole_cam.fifo 1280 720

clean:
	rm -rf build

calibrate_teams: src/capture/calibrate_teams.c
	$(CC) -std=c17 -O2 -o $@ $< -lm

detect_bags: src/capture/detect_bags.c
	$(CC) -std=c17 -O2 -o $@ $< -lm