# ASSIGNMENT_3
[![Python 3.6+](https://img.shields.io/badge/Python-3.6+-green.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-BrianAtkinson93-blueviolet?style=flat&logo=github)](https://github.com/BrianAtkinson93)
[![University of the Fraser Valley](https://img.shields.io/badge/University%20of%20the%20Fraser%20Valley-Visit-blue)](https://www.ufv.ca/)


---
Brian Atkinson<br>
300088157<br>
COMP_360 | Winter 2023<br>
Mon Feb 20, 2023<br>
Amir Shabani<br>

---
## Build and Run
This program is able to be run as a standalone binary or as a python script.<br>
Examples: (pressing the green arrow will execute all 4 commands)
```bash
python3 Part2.py -h
python3 Part2.py "Abbotsford" "West Vancouver" "A*"
python3 Part2.py "Abbotsford" "West Vancouver" Grassfire
python3 Part2.py "Abbotsford" "West Vancouver" Dijkstra
```
For building you can use the provided spec file and pyinstaller<br>
The binary will be placed into the /dist/ directory with name 'search'
```bash
pyinstaller search.spec
```
Execution:
```bash
./dist/search "Abbotsford" "West Vancouver" "A*"
```

## Assignment_Handout
### Part 1:

- Build a graph for the (offline) road map/network
  - Consider the following as the list of major cities in the greater Vancouver area and
  lower mainland:
    - Vancouver
    - North Vancouver
    - West Vancouver
    - Burnaby
    - Richmond
    - Surrey
    - New Westminster
    - Delta
    - Langley 
    - Abbotsford 
    - Chilliwack 
    - Hope 
    - Mission
    <br><br>
- Calculate the cost function (using Google map application)
  - Consider the minimum distance between the city halls (assume the mayor’s
office is in the city hall) as the cost of traveling between two cities. For simplicity, you could round down this 
number. For example, the cost of traveling from city hall Abbotsford to Chilliwack operation center= 32.7Km 
  - Note: if the city/district doesn’t have a city hall, search for a municipality building or downtown or ....
  <br><br>
- Build a heuristic function 
  - X= minimum distance between the city halls 
  - Y= a random number between 5 and 10 
  - H(c1,c2)=round_down (X-Y) # heuristic value between city 1 and 2
<br><br>
- Subtracting the random number (Y) from X in the heuristic function makes sure the heuristic value is admissible 
but doesn’t guarantee that it is consistent. Write a function that makes sure the heuristic values are consistent 
as well. This means, for some nodes the Y value should be changed to make their heuristic consistent.


### Part 2:

- Write a program that uses a graph-based path planning algorithm (i.e., A*) to guide the driverless car for traveling between two city halls. 
  - Write the function PathFinder where the input arguments to the function are the name of two cities and the search algorithm. 
    - Example: when you call the function PathFinder (VA, LA, A_Star), it returns the optimal path between Vancouver and Langley using A* algorithm.
    - (required) the function should implement the A* algorithm
    - (optional/bonus) extend your code to include Grassfire and Dijsktra’s
    algorithms as well.
  - Write your program with an intuitive user interaction parts for both taking the inputs
  from the user to providing the results.

