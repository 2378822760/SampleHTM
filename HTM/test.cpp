/**
 * @file 该文件用于测试整个HTM系统包括编码，时间池，空间池，以及解码
 * 用户可以自定义数据流用于测试系统
 */
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

void printFinalResult(ClassifierResult& result, const vector<Real64>& data, \
    UInt iteration, ScalarEncoder *encoder) {
    for (auto& step : result) {
        cout << step.first << "step predicate:" << endl;
        if (iteration >= step.first) {
            auto step_ = step.first;
            auto res = step.second.begin();
            auto real = data[iteration - step_];
            auto prediction = encoder->decode(max_element(res->begin(), res->end()) - 
                res->begin());
            cout << "real = " << real << ", prediction = " << prediction << endl;
        }
        else {
            cout << "No data." << endl;
        }
    }
}

void testParameter() {
// encoder
UInt W = 20;
Real64 MIN_VAL, MAX_VAL;
    Real64 BUCKET_SIZE = 0.2;
bool CLIP_INPUT = true;

// sp
    vector<UInt> COLUMN_DEMENSIONS = { 512 };
UInt ENCODE_SIZE;
// tm
UInt NUM_COLUMNS;

// sdrclassifier
vector<UInt> STEPS = { 1, 2, 3 };
Real64 LR = 0.02;

    vector<Real64> dataStream = getDataVector("1.txt");
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
        cout << "=====================================================" << endl;
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
        printFinalResult(result, dataStream, i, &encoder);
        cout << "=====================================================" << endl;
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

void testConfig() {
    config config_ = readConfig("config.txt");

    istream &inputStream = getInputStream(config_.at("InputStream"));

    ScalarEncoder encoder = ScalarEncoder(config_.at("ScalarEncoder"));

    UInt ENCODE_SIZE = encoder.getOutputWidth();
    config_["SpatialPooler"]["inputDimensions"] = to_string(ENCODE_SIZE);

    SpatialPooler sp(config_.at("SpatialPooler"));

    UInt NUM_COLUMNS = sp.numberOfColumns();
    config_["TemporalMemory"]["columnDimensions"] = to_string(NUM_COLUMNS);

    TemporalMemory tm(config_.at("TemporalMemory"));

    config_["SDRClassifier"]["cellsNum"]  = to_string(tm.numberOfCells());
    config_["SDRClassifier"]["bucketNum"] = to_string(encoder.bucketNum());
    
    SDRClassifier sdrclassifier(config_.at("SDRClassifier"));

    UInt* inputVector = (UInt*)calloc(ENCODE_SIZE, sizeof(UInt));
    UInt* columnVector = (UInt*)calloc(NUM_COLUMNS, sizeof(UInt));
    vector<Real64> data;
    Real64 currData;
    UInt i = 0;
    while (inputStream >> currData) {
        data.push_back(currData);
        cout << "=====================================================" << endl;
        cout << "current input : " << currData << endl;
        memset(inputVector, 0, sizeof(UInt) * ENCODE_SIZE);
        encoder.encodeIntoArray(currData, inputVector);
        sp.compute(inputVector, true, columnVector);
        vector<CellIndex> activeCells = tm.compute(NUM_COLUMNS, columnVector);
        ClassifierResult result;
        sdrclassifier.compute(activeCells, encoder.lastBucketIdx(), true, result);
        printFinalResult(result, data, i, &encoder);
        ++i;
        cout << "=====================================================" << endl;
    }
}

int main() {
    testConfig();
    // testParameter();
}