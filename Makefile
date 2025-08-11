CC=gcc
CFLAGS=-I./include -Wall
LDFLAGS=-lm

SRCS=src/matrix.c src/activation.c src/main.c
OBJS=$(SRCS:.c=.o)
TARGET=program

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)