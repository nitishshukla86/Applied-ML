#Dockerizing assignment 1   
Setup
1. Create docker image by running the command 
``` docker build -t assignment/ass2:1.0 . ``` 
inside ```Assignment1``` directory.

2. Run the image by executing
```docker run -it ass2 \bin\bash ```

3. Run ```python main.py``` to run the program inside container.
