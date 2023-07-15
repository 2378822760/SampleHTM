#include "Encoder.hpp"
#include "SpatialPooler.hpp"
#include "TemporalMemory.hpp"
#include "SDRClassifier.hpp"
#include "tdarray.hpp"
#include "HelpFunc.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <algorithm>
// #define DEBUG
using namespace std;
// encoder
UInt W = 20;
Real64 MIN_VAL, MAX_VAL;
Real64 BUCKET_SIZE = 0.1;
bool CLIP_INPUT = true;

// sp
vector<UInt> COLUMN_DEMENSIONS = { 1024 };
UInt ENCODE_SIZE;
// tm
UInt NUM_COLUMNS;

// sdrclassifier
vector<UInt> STEPS = { 1, 2, 3 };
Real64 LR = 0.02;

vector<Real64> getDataStream(const string& path);

int main() {
    vector<Real64> dataStream = getDataStream("1.txt");
    getArrayRange(dataStream.begin(), dataStream.end(), MIN_VAL, MAX_VAL);
    ScalarEncoder encoder = ScalarEncoder(W, MIN_VAL, MAX_VAL, \
        (MAX_VAL - MIN_VAL) / BUCKET_SIZE, CLIP_INPUT);
    ENCODE_SIZE = encoder.getOutputWidth();
    SpatialPooler sp({ ENCODE_SIZE }, COLUMN_DEMENSIONS);
    NUM_COLUMNS = sp.numberOfColumns();
    TemporalMemory tm({ NUM_COLUMNS });
    SDRClassifier sdrclassifier(STEPS, tm.numberOfCells(), \
        encoder.bucketNum(), LR);
    UInt* inputVector = (UInt*)calloc(ENCODE_SIZE, sizeof(UInt));
    UInt* columnVector = (UInt*)calloc(NUM_COLUMNS, sizeof(UInt));

    for (int i = 0; i < dataStream.size(); ++i) {
        cout << "iteration:" << i + 1 << ", current input: " \
            << dataStream[i] << endl;
        memset(inputVector, 0, sizeof(UInt) * ENCODE_SIZE);
        encoder.encodeIntoArray(dataStream[i], inputVector);
        sp.compute(inputVector, true, columnVector);
        vector<CellIndex> activeCells = tm.compute(NUM_COLUMNS, columnVector);
        ClassifierResult result;
        sdrclassifier.compute(activeCells, encoder.lastBucketIdx(), true, result);
#ifdef DEBUG
        cout << encoder.decode(encoder.lastBucketIdx()) << endl;
        printCodedDate(inputVector, inputVector + ENCODE_SIZE, ENCODE_SIZE);
        printCodedDate(columnVector, columnVector + NUM_COLUMNS, NUM_COLUMNS / 2);
        cout << "TM--activeCells's size = " << activeCells.size() << '\n';
        sdrclassifier.logResult(result);
#endif // DEBUG
        for (auto& step : result) {
            cout << step.first << "step predicate:" << endl;
            if (i >= step.first) {
                auto step_ = step.first;
                auto res = step.second.begin();
                auto real = dataStream[i - step_];
                auto prediction = encoder.decode(max_element(res->begin(), res->end()) - res->begin());
                cout << "real = " << real << ", prediction = " << prediction << endl;
            }
            else {
                cout << "No data." << endl;
            }
        }
    }


}

vector<Real64> getDataStream(const string& path) {
    ifstream inputData(path);
    if (!inputData.is_open()) {
        cerr << "Can't open file!" << endl;
    }
    Real64 data;
    vector<Real64> dataStream;
    while (inputData >> data) {
        dataStream.push_back(data);
    }
    if (inputData.eof())
        cout << "End of file reached.\n";
    else if (inputData.fail())
        cout << "Input terminated by data mismatch.\n";
    else
        cout << "Input terminated for unknown reason.\n";
    cout << "Data size : " << dataStream.size() << endl;
    inputData.close();
    return dataStream;
}

int test() {
	/* Input */
	double data[] = { 10,10.5,11,11.5,12,12.5,13,14,15,16,17,18,19 };

	/* Encoder Part */
	int arraySize = sizeof(data) / sizeof(double);
	cout << "arraySize is " << arraySize << '\n';
	double minElement, maxElement;
	getArrayRange(data, data + arraySize, minElement, maxElement);
	ScalarEncoder encoder = ScalarEncoder(5, 10, 20, 20, true);
	UInt output_width = encoder.getOutputWidth();
	cout << "Encoder--output_width:" << output_width << endl;

	/*  Class Declaration */
	SpatialPooler sp({output_width}, {100});
	const auto numColumns = sp.numberOfColumns();
	cout << "SP--Total amount of columns is " << numColumns << '\n';
	TemporalMemory tm({ numColumns });
	cout << "TM--Total amount of cells is " << tm.numberOfCells() << '\n';
	SDRClassifier sdrclassifier({1, 3}, tm.numberOfCells(), encoder.bucketNum(), 0.02);

	UInt* inputVector = (UInt*)calloc(output_width, sizeof(UInt));
	UInt* columnVector = (UInt*)calloc(numColumns, sizeof(UInt));

	for (int i = 0; i < arraySize; ++i) {
		cout << "iteration:" << i + 1 << ", current input: " << data[i] << endl;
		memset(inputVector, 0, sizeof(UInt)*output_width);
		encoder.encodeIntoArray(data[i], inputVector);
		sp.compute(inputVector, true, columnVector);
		vector<CellIndex> activeCells = tm.compute(numColumns, columnVector);
		ClassifierResult result;
		sdrclassifier.compute(activeCells, encoder.lastBucketIdx(), true, result);
#ifdef DEBUG
		cout << encoder.decode(encoder.lastBucketIdx()) << endl;
		printCodedDate(inputVector, inputVector + output_width, output_width);
		printCodedDate(columnVector, columnVector + numColumns, numColumns / 2);
		cout << "TM--activeCells's size = " << activeCells.size() << '\n';
		sdrclassifier.logResult(result);
#endif // DEBUG
		for (auto& step : result) {
			cout << step.first << "step predicate:" << endl;
			if (i >= step.first) {
				auto step_ = step.first;
				auto res = step.second.begin();
				auto real = data[i - step_];
				auto prediction = encoder.decode(max_element(res->begin(), res->end()) - res->begin());
				cout << "real = " << real << ", prediction = " << prediction << endl;
			}
			else {
				cout << "No data." << endl;
			}
		}
	}
    return 0;
}