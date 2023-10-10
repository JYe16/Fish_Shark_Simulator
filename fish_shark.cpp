#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */

/*
 HPC Semester Project: Fish and Shark
 Author: Jackie Ye, Wynn Pho
 Input: input.txt
 Input format: number of rows\n, number of columns\n,
            number of fishes\n, number of sharks\n, breeding age for fishes\n,
            breeding age for sharks\n, starvation time for sharks\n
            time of period\n, direction of current
 How to compile: mpicxx -Wall -o fish_shark fish_shark.cpp
 */

using namespace std;

//define objects
const char EMPTY = '.';
const char OBJECT_FISH = 'f';
const char OBJECT_SHARK = 'S';
const char GHOST = '*';

//define variables that will not change
const int FISH_GENERATION = 20;
const int SHARK_GENERATION = 40;
const int PROC = 0;
const double BIAS_VALUE = 0.2;
const string BLUE("\033[0;34m");
const string GREEN("\033[0;32m");
const string RESET("\033[0m");
const string RED("\033[0;31m");


int current;

MPI_Status status;

struct cell {
    char status;
    int b_age;
    int starv_time;
    int gen;
    bool moved;
    bool changed;
};

void initOcean(cell **ocean, int rows, int cols, int rank, int chunkSize, int numFish, int numShark, int fishAge, int sharkAge, int starvationTime);
void firstCommunication(cell **ocean, int rank, int size, int cols, int chunkSize, MPI_Datatype cell_type);
void secondCommunication(cell **ocean, int rank, int size, int cols, int chunkSize, MPI_Datatype cell_type);
void remove(cell **object, int i, int j);
void simulation(cell **ocean, int rank, int size, int cols, int chunkSize, int fishAge, int sharkAge, int starvationTime);
void move(cell **ocean, int i, int j, int x, int y, int age);
void fishBreed(cell **ocean, int i, int j, int fishAge);
void sharkBreed(cell **ocean, int i, int j, int sharkAge, int starvationTime);
bool randBool(double bias);


int main(int argc, char *argv[]) {

    //declaring variables
    string line;
    int rows;
    int cols;
    int totalFish;
    int totalShark;
    int numFish;
    int numShark;
    int fishAge;
    int sharkAge;
    int starvationTime;
    int rank, size;
    int chunkSize;
    int totalTime;

    double t1, t2, time;

    //fread inputs from file
    ifstream infile;
    infile.open("input2.txt"); 
    getline(infile, line);
    rows = stoi(line);
    getline(infile, line);
    cols = stoi(line);
    getline(infile, line);
    totalFish = stoi(line);
    getline(infile, line);
    totalShark = stoi(line);
    getline(infile, line);
    fishAge = stoi(line);
    getline(infile, line);
    sharkAge = stoi(line);
    getline(infile, line);
    starvationTime = stoi(line);
    getline(infile, line);
    totalTime = stoi(line);
    getline(infile, line);
    current = stoi(line);

    MPI_Init(&argc, &argv);

    struct cell _cell;
    int count = 3;
    MPI_Datatype array_of_types[count] = {MPI_CHAR, MPI_INT, MPI_C_BOOL};
    int array_of_blocklengths[count] = {1, 3, 2};

    MPI_Aint array_of_displacements[count];
    MPI_Aint address1, address2, address3, address4;
    
    MPI_Get_address(&_cell, &address1);
    MPI_Get_address(&_cell.status, &address2);
    MPI_Get_address(&_cell.b_age, &address3);
    MPI_Get_address(&_cell.moved, &address4);

    array_of_displacements[0] = address2 - address1;
    array_of_displacements[1] = address3 - address1;
    array_of_displacements[2] = address4 - address1;

    MPI_Datatype cell_type;
    MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, array_of_types, &cell_type);
    MPI_Type_commit(&cell_type);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((rows * cols) < (totalFish + totalShark))
    {
        if(rank == PROC){
            cout << "TOO MANY FISHES!" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    if(!(current >= 0 && current <= 3)){
        if(rank == PROC){
            cout << "INVALID CURRENT!" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    chunkSize = rows / size;
    if (rows % size != 0)
    {
        for (int i = 1; i < (rows % size + 1); i++)
        {
            if (rank == i)
            {
                chunkSize += 1;
            }
        }
    }

    numFish = totalFish / size;
    numShark = totalShark / size;
    if (totalFish % size != 0)
    {
        for (int i = 1; i < (totalFish % size + 1); i++)
        {
            if (rank == i)
            {
                numFish += 1;
            }
        }
    }
    if (totalShark % size != 0)
    {
        for (int i = 1; i < (totalShark % size + 1); i++)
        {
            if (rank == i)
            {
                numShark += 1;
            }
        }
    }

    //array declaration
    cell **ocean = (cell **)malloc((chunkSize + 2) * sizeof(cell *));

    for (int i = 0; i < chunkSize + 2; i++)
    {
        ocean[i] = (cell *)malloc(cols * sizeof(cell));
    }

    initOcean(ocean, rows, cols, rank, chunkSize, numFish, numShark, fishAge, sharkAge, starvationTime);

    if(rank == PROC){
        cout << "\n" << "FISH-SHARK SIMULATION: " << endl;
        cout << "SIZE OF OCEAN: " << rows << " x " << cols << endl;
        cout << "SIZE OF SUB-OCEAN: " << chunkSize << " x " << cols << endl;
        cout << "NUMBER OF FISHES: " << totalFish << endl;
        cout << "NUMBER OF SHARK: " << totalShark << endl;
        if(current == 0){
            cout << "CURRENT: NORTH" << endl;
        }else if(current == 1){
            cout << "CURRENT: SOUTH" << endl;
        }else if(current == 2){
            cout << "CURRENT: EAST" << endl;
        }else{
            cout << "CURRENT: WEST" << endl;
        }
        usleep(3000000);
        system("clear");
    }

    if (rank == 0)  t1 = MPI_Wtime();  // start time

    for (int numTime = 0; numTime < totalTime; numTime++)
    {
        /* First round of communication */
        firstCommunication(ocean, rank, size, cols, chunkSize, cell_type);

        //system("clear");
        /* For testing */
        if (rank == PROC)
        {
            cout << "PERIOD: " << numTime << endl;
            for (int i = 2; i < chunkSize; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (ocean[i][j].status == OBJECT_FISH) 
                        cout << GREEN << ocean[i][j].status << RESET << " ";
                    else if (ocean[i][j].status == OBJECT_SHARK) 
                        cout << RED << ocean[i][j].status << RESET << " ";
                    else cout << ocean[i][j].status << " ";    
                }
                cout << endl;
            }
            cout << endl;
            usleep(100000); // microsecs - wynn changed here for testing ()
            system("clear");
        }


        simulation(ocean, rank, size, cols, chunkSize, fishAge, sharkAge, starvationTime);

        /* Second round of communication */
        secondCommunication(ocean, rank, size, cols, chunkSize, cell_type);

        for (int i = 1; i < chunkSize + 1; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                ocean[i][j].moved = false;
                ocean[i][j].changed = false;
            }
        }
    }

    if (rank == 0) {
        t2 = MPI_Wtime(); // finish time
        time = t2 - t1 - 0.1 * totalTime;
        ofstream fileOUT("output_10.txt", ios::app);
        fileOUT << size << "," << time << endl;
        fileOUT.close();
    }
    free(ocean);
    MPI_Type_free(&cell_type);

    MPI_Finalize();
}

void remove(cell **object, int i, int j) {
    object[i][j].status = EMPTY;
    object[i][j].b_age = -1;
    object[i][j].starv_time = -1;
    object[i][j].gen = -1;
}

void initOcean(cell **ocean, int rows, int cols, int rank, int chunkSize, int numFish, int numShark, int fishAge, int sharkAge, int starvationTime) {
    for (int i = 0; i < chunkSize + 2; i++) {
        for (int j = 0; j < cols; j++) {
            ocean[i][j].status = GHOST;
            //set moved to false
            ocean[i][j].moved = false;
            //set the bredding age to -1
            ocean[i][j].b_age = -1;
            //set the genration number to -1
            ocean[i][j].gen = -1;
            //set changed to false
            ocean[i][j].changed = false;
            //set the starvation time for sharks to -1
            ocean[i][j].starv_time = -1;
        }
    }

    for (int i = 1; i < chunkSize + 1; i++) {
        for (int j = 0; j < cols; j++) {
            ocean[i][j].status = EMPTY;
        }
    }

    srand(time(NULL) + rank); // init random seed
    int x;
    int y;
    while (numFish > 0) {
        x = (rand() % (chunkSize + 1) + 1);
        y = rand() % cols;
        if (ocean[x][y].status == EMPTY)
        {
            ocean[x][y].status = OBJECT_FISH;
            ocean[x][y].b_age = fishAge;
            ocean[x][y].gen = FISH_GENERATION;
            numFish--;
        }
    }

    while (numShark > 0) {
        x = (rand() % (chunkSize + 1) + 1);
        y = rand() % cols;
        if (ocean[x][y].status == EMPTY) {
            ocean[x][y].status = OBJECT_SHARK;
            ocean[x][y].b_age = sharkAge;
            ocean[x][y].starv_time = starvationTime;
            ocean[x][y].gen = SHARK_GENERATION;
            numShark--;
        }
    }
}

void firstCommunication(cell **ocean, int rank, int size, int cols, int chunkSize, MPI_Datatype cell_type) {
    cell cellRecv;
    cell cellSend;
    if (rank == size - 1) {
        for (int i = 0; i < cols; i++) {
            //for the last process, send the first row (1) to the last row of previous process
            cellSend = ocean[1][i];
            MPI_Send(&cellSend, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD);
            //receive the last row from the previous process and paste to the top ghost row (0)
            MPI_Recv(&cellRecv, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ocean[0][i] = cellRecv;
        }
    }
    else if (rank == 0) {
        for (int i = 0; i < cols; i++) {
            //for process #0, send the last row (chunkSize) to the first row of process #1
            cellSend = ocean[chunkSize][i];
            MPI_Send(&cellSend, 1, cell_type, 1, 0, MPI_COMM_WORLD);
            //receive the first row from the process #1 and paste to the bottom ghost row (chunkSize + 1)
            MPI_Recv(&cellRecv, 1, cell_type, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ocean[chunkSize + 1][i] = cellRecv;
        }
    } else {
        for (int i = 0; i < cols; i++) {
            //send row #1 to rank - 1
            cellSend = ocean[1][i];
            MPI_Send(&cellSend, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD);
            //receive from rank - 1 and paste them to top ghost row
            MPI_Recv(&cellRecv, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ocean[0][i] = cellRecv;
        }
        for (int i = 0; i < cols; i++) {
            //send the last row (ChunkSize) to rank + 1
            cellSend = ocean[chunkSize][i];
            MPI_Send(&cellSend, 1, cell_type, rank + 1, 0, MPI_COMM_WORLD);
            //receive from rank + 1 and paste them to bottom ghost row
            MPI_Recv(&cellRecv, 1, cell_type, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ocean[chunkSize + 1][i] = cellRecv;
        }
    }
}

void secondCommunication(cell **ocean, int rank, int size, int cols, int chunkSize, MPI_Datatype cell_type) {
    cell cellRecv;
    cell cellSend;
    if (rank == size - 1) {
        for (int i = 0; i < cols; i++) {
            //for the last process, send the top ghost row (0) to the last row of previous process
            cellSend = ocean[0][i];
            MPI_Send(&cellSend, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD);
            //receive the bottom ghost row from the previous process and paste to the first row (1)
            MPI_Recv(&cellRecv, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD, &status);

            if(ocean[1][i].changed == false){
                ocean[1][i] = cellRecv;
            }
        }
    }
    else if (rank == 0) {
        for (int i = 0; i < cols; i++) {
            //for process #0, send the bottom row (chunkSize + 1) to the first row of process #1
            cellSend = ocean[chunkSize + 1][i];
            MPI_Send(&cellSend, 1, cell_type, 1, 0, MPI_COMM_WORLD);
            //receive the top ghost row from the process #1 and paste to the last row (chunkSize)
            MPI_Recv(&cellRecv, 1, cell_type, 1, 0, MPI_COMM_WORLD, &status);

            if(ocean[chunkSize][i].changed == false){
                ocean[chunkSize][i] = cellRecv;
            }
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            //send top ghost row (0) to the last row rank - 1
            cellSend = ocean[0][i];
            MPI_Send(&cellSend, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD);
            //receive from rank - 1 and paste them to the first row (1)
            MPI_Recv(&cellRecv, 1, cell_type, rank - 1, 0, MPI_COMM_WORLD, &status);
            
            if(ocean[1][i].changed == false){
                ocean[1][i] = cellRecv;
            }

        }
        for (int i = 0; i < cols; i++)
        {
            //send the bottom ghost row (ChunkSize + 1) to rank + 1
            cellSend = ocean[chunkSize + 1][i];
            MPI_Send(&cellSend, 1, cell_type, rank + 1, 0, MPI_COMM_WORLD);
            //receive from rank + 1 and paste them to the last row (chunkSize)
            MPI_Recv(&cellRecv, 1, cell_type, rank + 1, 0, MPI_COMM_WORLD, &status);
            
            if(ocean[chunkSize][i].changed == false){
                ocean[chunkSize][i] = cellRecv;
            }
        }
    }
}

void simulation(cell **ocean, int rank, int size, int cols, int chunkSize, int fishAge, int sharkAge, int starvationTime) {
    vector<int> fishDirection;
    vector<char> neighborVector;
    vector<int> emptyDirection;
    vector<int>::iterator it;
    bool biased;
    int index;
    for (int i = 1; i < chunkSize + 1; i++) {
        for (int j = 0; j < cols; j++) {
            //subtract 1 from generation
            //generation(ocean, chunkSize, cols); 

            //clear all vectors
            neighborVector.clear();
            fishDirection.clear();
            emptyDirection.clear();

            neighborVector.push_back(ocean[i - 1][j].status);
            neighborVector.push_back(ocean[i + 1][j].status);
            neighborVector.push_back(ocean[i][j + 1].status);
            neighborVector.push_back(ocean[i][j - 1].status);

            srand(time(0) + j + 100 * i);
            if (ocean[i][j].status == OBJECT_SHARK){
                // k: 0: north, 1: south, 2: east, 3: west
                //shark dies if age = 0
                if(ocean[i][j].starv_time == 0 || ocean[i][j].gen == 0){
                    remove(ocean, i, j);
                }

                else if (ocean[i][j].moved == false){
                    for (unsigned k = 0; k < neighborVector.size(); k++) {
                        if (neighborVector[k] == OBJECT_FISH) {
                            fishDirection.push_back(k);
                        } else if (neighborVector[k] == EMPTY) {
                            emptyDirection.push_back(k);
                        }
                    }

                    if (fishDirection.size() != 0)
                    { // if there's a fish in any adjacenct cell
                        it = find(fishDirection.begin(), fishDirection.end(), current);
                        if(it != fishDirection.end()){
                            biased = randBool(BIAS_VALUE);
                            if(biased == true){
                                index = current;
                            }else{
                                index = rand() % fishDirection.size();
                            }
                        }else{
                            index = rand() % fishDirection.size();
                        }

                        // k: 0: north, 1: south, 2: east, 3: west
                        if (fishDirection[index] == 0) {
                            move(ocean, i, j, i - 1, j, sharkAge);
                        }
                        else if (fishDirection[index] == 1) {
                            move(ocean, i, j, i + 1, j, sharkAge);
                        } else if (fishDirection[index] == 2) {
                            move(ocean, i, j, i, j + 1, sharkAge);
                        } else if (fishDirection[index] == 3) {
                            move(ocean, i, j, i, j - 1, sharkAge);
                        }
                        remove(ocean, i, j);
                    }
                    else if (emptyDirection.size() != 0)
                    {

                        ocean[i][j].starv_time -= 1;

                        for (unsigned k = 0; k < neighborVector.size(); k++)
                        {
                            if (neighborVector[k] == EMPTY)
                            {
                                emptyDirection.push_back(k);
                            }
                        }

                        it = find(emptyDirection.begin(), emptyDirection.end(), current);
                        if(it != emptyDirection.end()){
                            biased = randBool(BIAS_VALUE);
                            if(biased == true){
                                index = current;
                            }else{
                                index = rand() % emptyDirection.size();
                            }
                        }else{
                            index = rand() % emptyDirection.size();
                        }
                        // k: 0: north, 1: south, 2: east, 3: west
                        if (emptyDirection[index] == 0)
                        {
                            move(ocean, i, j, i - 1, j, sharkAge);
                        }
                        else if (emptyDirection[index] == 1)
                        {
                            move(ocean, i, j, i + 1, j, sharkAge);
                        }
                        else if (emptyDirection[index] == 2)
                        {
                            move(ocean, i, j, i, j + 1, sharkAge);
                        }
                        else if (emptyDirection[index] == 3)
                        {
                            move(ocean, i, j, i, j - 1, sharkAge);
                        }
                        if(ocean[i][j].b_age == 0){
                            sharkBreed(ocean, i, j, sharkAge, starvationTime);
                        }else{
                            remove(ocean, i, j);
                        }
                    }
                    else
                    {
                        ocean[i][j].gen -= 1;
                        ocean[i][j].b_age -= 1;
                        ocean[i][j].starv_time -= 1;
                        ocean[i][j].changed = true;
                    }
                }
            }
            else if (ocean[i][j].status == OBJECT_FISH)
            {
                if(ocean[i][j].gen == 0){
                    remove(ocean, i, j);
                }else if(ocean[i][j].moved == false){
                    for (unsigned k = 0; k < neighborVector.size(); k++)
                    {
                        if (neighborVector[k] == EMPTY)
                        {
                            emptyDirection.push_back(k);
                        }
                    }
                    

                    if (emptyDirection.size() != 0)
                    {

                        for (unsigned k = 0; k < neighborVector.size(); k++)
                        {
                            if (neighborVector[k] == EMPTY)
                            {
                                emptyDirection.push_back(k);
                            }
                        }

                        it = find(emptyDirection.begin(), emptyDirection.end(), current);
                        if(it != emptyDirection.end()){
                            biased = randBool(BIAS_VALUE);
                            if(biased == true){
                                index = current;
                            }else{
                                index = rand() % emptyDirection.size();
                            }
                        }else{
                            index = rand() % emptyDirection.size();
                        }

                        // k: 0: north, 1: south, 2: east, 3: west
                        if (emptyDirection[index] == 0){
                            move(ocean, i, j, i - 1, j, fishAge);
                        } else if (emptyDirection[index] == 1) {
                            move(ocean, i, j, i + 1, j, fishAge);
                        } else if (emptyDirection[index] == 2) {
                            move(ocean, i, j, i, j + 1, fishAge);
                        } else if (emptyDirection[index] == 3) {
                            move(ocean, i, j, i, j - 1, fishAge);
                        }

                        if(ocean[i][j].b_age == 0){
                            fishBreed(ocean, i, j, fishAge);
                        }else{
                            remove(ocean, i, j);
                        }
                    } else {
                        ocean[i][j].gen -= 1;
                        ocean[i][j].changed = true;
                        ocean[i][j].b_age -= 1;
                    }
                }
            }
        }
    }
}

void move(cell **ocean, int i, int j, int x, int y, int age) {
    ocean[x][y].status = ocean[i][j].status;
    ocean[x][y].gen = ocean[i][j].gen;
    ocean[x][y].gen -= 1;
    ocean[x][y].moved = true;
    ocean[x][y].changed = true;
    ocean[i][j].changed = true;
    ocean[x][y].starv_time = ocean[i][j].starv_time;

    if(ocean[i][j].b_age != 0){
        ocean[x][y].b_age = ocean[i][j].b_age - 1;
    }else{
        ocean[x][y].b_age = age;
    }
}

void fishBreed(cell **ocean, int i, int j, int fishAge) {
    ocean[i][j].status = OBJECT_FISH;
    ocean[i][j].b_age = fishAge;
    ocean[i][j].gen = FISH_GENERATION;
    ocean[i][j].starv_time = -1;
}

void sharkBreed(cell **ocean, int i, int j, int sharkAge, int starvationTime) {
    ocean[i][j].status = OBJECT_SHARK;
    ocean[i][j].b_age = sharkAge;
    ocean[i][j].gen = SHARK_GENERATION;
    ocean[i][j].starv_time = starvationTime;
}

bool randBool(double bias) {
    return rand() < ((RAND_MAX + 1.0) * ((bias + 1) / 2));
}
