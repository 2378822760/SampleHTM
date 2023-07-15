#pragma once
#include "type.hpp"
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
using namespace std;
/**
 * ����arr�����½磬�Ͻ籣����max���½籣����min
 */
template<typename T, typename iterationType>
void getArrayRange(iterationType beg, iterationType end, T& min, T& max) {
	min = max = *beg;
	size_t arraySize = 0;
	for (iterationType it = beg; it != end; it++) {
		arraySize++;
		T tmp = *it;
		min = min > tmp ? tmp : min;
		max = max < tmp ? tmp : max;
	}
	max += (max - min) / (arraySize - 1);
}

/**
 * ���д�ӡԪ�أ�ÿ�д�ӡn��Ԫ��
 */
template<typename iterationType>
void printCodedDate(iterationType beg, iterationType end, UInt n) {
	UInt cnt = 0;
	for (iterationType it = beg; it != end; it++) {
		cout << *it << ' ';
		cnt++;
		if (cnt % n == 0) cout << '\n';
	}
}

/**
 * ��ȡ�����ļ�
 */
typedef  map<string, map<string, string>> config;
config readConfig(const string& path) {
    ifstream config_file(path);
    config config_data;
    string current_section;

    if (config_file.is_open()) {
        string line;
        while (getline(config_file, line)) {
            // ���Կ��к�ע����
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // ����Ƿ�Ϊ�µĲ���
            if (line[0] == '[' && line[line.length() - 1] == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }

            // ������ֵ��
            size_t equals_pos = line.find('=');
            if (equals_pos != string::npos) {
                std::string key = line.substr(0, equals_pos);
                std::string value = line.substr(equals_pos + 1);

                // ȥ����β�ո�
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                config_data[current_section][key] = value;
            }
        }
    }
    else {
        cerr << "Can't open file " << path << endl;
    }

    config_file.close();
    return config_data;
}