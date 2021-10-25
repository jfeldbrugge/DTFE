all: compile

compile:
	pandoc index.md --css css/style.css -s --mathjax -t html5 -o index.html

clean: 
	rm index.html

.PHONY: compile clean