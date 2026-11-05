iirquant: iirquant.c
	gcc iirquant.c -o iirquant -lm

clean:
	rm -f iirquant *.o
