#include <stdio.h>
#include <stdlib.h>
//#include <iostream>
#include <time.h>
#include <math.h>
#include <mpi.h>

//using namespace std;

int ProcNum = 0;      // Number of available processes 
int ProcRank = 0;     // Rank of current process

// Function for random initialization of matrix elements
void RandomDataInitialization(double* pAMatrix, double* pBMatrix,
	int Size) {
	int i, j;  // Loop variables
	srand(unsigned(clock()));
	for (i = 0; i < Size; i++)
		for (j = 0; j < Size; j++) {
			pAMatrix[i*Size + j] = rand() / double(1000);
			pBMatrix[i*Size + j] = rand() / double(1000);
		}
}

// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
	int i, j; // Loop variables
	for (i = 0; i < RowCount; i++) {
		for (j = 0; j < ColCount; j++)
			printf("%7.4f ", pMatrix[i*ColCount + j]);
		printf("\n");
	}
}

// Function for matrix multiplication
void SerialResultCalculation(double* pAMatrix, double* pBMatrix,
	double* pCMatrix, int Size) {
	int i, j, k;  // Loop variables
	for (i = 0; i < Size; i++) {
		for (j = 0; j < Size; j++)
			for (k = 0; k < Size; k++)
				pCMatrix[i*Size + j] += pAMatrix[i*Size + k] * pBMatrix[k*Size + j];
	}
}

// Function for block multiplication
void RowsMultiplication(double* pArows, double* pBrows,
	double* pCrows, int Size, int RowNum, int index) {
	//int i, j, k;  // Loop variables
	//for (i = 0; i < RowNum; i++) {
	//	for (j = 0; j < Size; j++)
	//		pCrows[i*Size + j] += pArows[i*Size + j] * pBrows[i*Size + j];
	//}
	int i, j, k;  // Loop variables
	for (i = 0; i < RowNum; i++) {
		for (j = 0; j < Size; j++)
			for (k = 0; k < RowNum; k++)
				pCrows[i*Size + j] += pArows[i*Size + index * RowNum + k] * pBrows[k*Size + j];
	}
}


// Function for memory allocation and data initialization
void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, double* &pArows,
	double* &pBrows, double* &pCrows, int &Size, int &RowNum) {
	/*if (ProcRank == 0) {
		do {
			printf("\nEnter size of the initial objects: ");
			scanf_s("%d", &Size);
		} while (Size < 0);
	}*/
	int RestRows; // Number of rows, that haven’t been distributed yet
	int i;             // Loop variable

	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);


	// Determine the number of matrix rows stored on each process
	RestRows = Size;
	for (i = 0; i < ProcRank; i++)
		RestRows = RestRows - RestRows / (ProcNum - i);
	RowNum = RestRows / (ProcNum - ProcRank);

	pArows = new double[RowNum*Size];
	pBrows = new double[RowNum*Size];
	pCrows = new double[RowNum*Size];

	for (int i = 0; i < RowNum*Size; i++) {
		pCrows[i] = 0;
	}
	pCMatrix = new double[Size*Size];

	if (ProcRank == 0) {
		pAMatrix = new double[Size*Size];
		pBMatrix = new double[Size*Size];
		//DummyDataInitialization(pAMatrix, pBMatrix, Size);
		RandomDataInitialization(pAMatrix, pBMatrix, Size);
	}
}


// Data distribution among the processes
void DataDistribution(double* pAMatrix, double* pBMatrix, double* pArows, double* pBrows, int Size, int RowNum) {

	int *pSendNum; // The number of elements sent to the process
	int *pSendInd; // The index of the first data element sent to the process
	int RestRows = Size; // Number of rows, that haven’t been distributed yet

	// Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];

	// Define the disposition of the matrix rows for current process
	RowNum = (Size / ProcNum);
	pSendNum[0] = RowNum * Size;
	pSendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestRows -= RowNum;
		RowNum = RestRows / (ProcNum - i);
		pSendNum[i] = RowNum * Size;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
	}

	// Scatter the A rows
	MPI_Scatterv(pAMatrix, pSendNum, pSendInd, MPI_DOUBLE, pArows, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Scatter the B rows
	MPI_Scatterv(pBMatrix, pSendNum, pSendInd, MPI_DOUBLE, pBrows, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;

}

// Function for gathering the result matrix
void ResultCollection(double* pCMatrix, double* pCrows, int Size,
	int RowNum) {

	int i;             // Loop variable
	int *pReceiveNum;  // Number of elements, that current process sends
	int *pReceiveInd;  /* Index of the first element from current process
						  in result vector */
	int RestRows = Size * Size; // Number of rows, that haven’t been distributed yet

	//Alloc memory for temporary objects
	pReceiveNum = new int[ProcNum];
	pReceiveInd = new int[ProcNum];

	//Define the disposition of the result vector block of current processor
	pReceiveInd[0] = 0;
	pReceiveNum[0] = Size * RowNum;
	for (i = 1; i < ProcNum; i++) {
		RestRows -= pReceiveNum[i - 1];
		pReceiveNum[i] = RestRows / (ProcNum - i);
		pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
	}

	//Gather the whole result vector on every processor
	MPI_Allgatherv(pCrows, pReceiveNum[ProcRank], MPI_DOUBLE, pCMatrix,
		pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);

	//Free the memory
	delete [] pReceiveNum;
	delete [] pReceiveInd;
}


// Cyclic shift of matrix B blocks in the process grid columns 
void BRowsCommunication(double *pBrows, int Size, int RowNum) {
	MPI_Status Status;
	int NextProc = ProcRank + 1;
	if (ProcRank == ProcNum - 1) NextProc = 0;
	int PrevProc = ProcRank - 1;
	if (ProcRank == 0) PrevProc = ProcNum - 1;

	MPI_Sendrecv_replace(pBrows, Size*RowNum, MPI_DOUBLE,
		NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
}

void ParallelResultCalculation(double* pArows,
	double* pBrows, double* pCrows, int Size, int RowNum) {

	int index = 0;
	for (int iter = 0; iter < ProcNum; iter++) {
		if (iter == 0) index = ProcRank;
		else {
			index--;
			if (index == -1)
				index = ProcNum - 1;
		}
		// multiplication
		RowsMultiplication(pArows, pBrows, pCrows, Size, RowNum, index);
		// Cyclic shift of rows of B   
		BRowsCommunication(pBrows, Size, RowNum);
	}
}


void TestResult(double* pAMatrix, double* pBMatrix, double* pCMatrix,
	int Size) {
	double* pSerialResult;     // Result matrix of serial multiplication
	double Accuracy = 1.e-6;   // Comparison accuracy
	int equal = 0;             // =1, if the matrices are not equal
	int i;                     // Loop variable

	if (ProcRank == 0) {
		pSerialResult = new double[Size*Size];
		for (i = 0; i < Size*Size; i++) {
			pSerialResult[i] = 0;
		}
		SerialResultCalculation(pAMatrix, pBMatrix, pSerialResult, Size);
		for (i = 0; i < Size*Size; i++) {
			if (fabs(pSerialResult[i] - pCMatrix[i]) >= Accuracy)
				equal = 1;
		}
		if (equal == 1)
			printf("The results of serial and parallel algorithms are NOT identical. Check your code.");
		else
			printf("The results of serial and parallel algorithms are identical.");
	}
}

// Function for computational process termination
void ProcessTermination(double* pAMatrix, double* pBMatrix,
	double* pCMatrix, double* pAblock, double* pBblock, double* pCblock) {
	if (ProcRank == 0) {
		delete [] pAMatrix;
		delete [] pBMatrix;
		delete [] pCMatrix;
	}
	delete [] pAblock;
	delete [] pBblock;
	delete [] pCblock;
}

void main(int argc, char* argv[]) {
	double* pAMatrix;  // The first argument of matrix multiplication
	double* pBMatrix;  // The second argument of matrix multiplication
	double* pCMatrix;  // The result matrix
	int Size;          // Size of matricies
	double *pArows;   // Initial block of matrix A on current process
	double *pBrows;   // Initial block of matrix B on current process
	double *pCrows;   // Block of result matrix C on current process
	int RowNum;          // Number of rows in the matrix  A stripe
	double Start, Finish, Duration;

	int N[] = { 100, 500, 1000, 1500, 2000,2500,3000 };


	setvbuf(stdout, 0, _IONBF, 0);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	for (int i = 0; i < 7; i++) {
		Size = N[i];
	/*if (ProcRank == 0)
	{
		getchar();
	}
	MPI_Barrier(MPI_COMM_WORLD);*/

	if (ProcRank == 0)
		printf("Parallel matrix multiplication program\n");

	// Memory allocation and initialization of matrix elements
	ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, pArows, pBrows,
		pCrows, Size, RowNum);


	Start = MPI_Wtime();
	DataDistribution(pAMatrix, pBMatrix, pArows, pBrows, Size,
		RowNum);


	// Execution 
	ParallelResultCalculation(pArows, pBrows,
		pCrows, Size, RowNum);

	ResultCollection(pCMatrix, pCrows, Size, RowNum);
	Finish = MPI_Wtime();

	Duration = Finish - Start;
	TestResult(pAMatrix, pBMatrix, pCMatrix, Size);

	if (ProcRank == 0) {
		printf("Time of execution = %f\n", Duration);
	}
	// Process Termination
	ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pArows, pBrows,
		pCrows);



	}
	MPI_Finalize();


}
