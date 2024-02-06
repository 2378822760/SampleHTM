#include "HelpFunc.hpp"

// #define DEBUG
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
        throw runtime_error("Can't open file!");
    }

    config_file.close();
    return config_data;
}

vector<Real64> getDataVector(const string& path) {
    ifstream inputData(path);
    if (!inputData.is_open()) {
        cerr << "Can't open file!" << endl;
        throw runtime_error("Can't open file!");
    }
    Real64 data;
    vector<Real64> dataVector;
    while (inputData >> data) {
        dataVector.push_back(data);
    }
    if (inputData.eof()) {
        cout << "End of file reached." << endl;
    }
    else if (inputData.fail()) {
        cerr << "Input terminated by data mismatch." << endl;
        throw runtime_error("Input terminated by data mismatch.");

    }
    else {
        cerr << "Input terminated for unknown reason." << endl;
        throw runtime_error("Input terminated for unknown reason.");
    }
#ifdef DEBUG
    cout << "Data size : " << dataStream.size() << endl;
#endif
    inputData.close();
    return dataVector;
}

istream& getInputStream(const map<string, string>& config)
{
    if (config.at("flag") == "true")
    {
        // ��ָ���ļ���������
        static ifstream inputFileStream(config.at("path"));
        return inputFileStream;
    }
    else
    {
        // ���ر�׼������
        return cin;
    }
}

vector<UInt> stov(const string& str) {
    vector<UInt> result;
    int i = 0, n = str.length();
    while (i < n) {
        while (i < n && !isdigit(str[i])) i++;
        if (i >= n) break;
        UInt tmp = 0;
        while (i < n && isdigit(str[i])) {
            tmp = tmp * 10 + str[i] - '0';
            i++;
        }
        result.push_back(tmp);
    }
    return result;
}