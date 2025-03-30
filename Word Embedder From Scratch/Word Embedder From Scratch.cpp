#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "model.h"

unordered_map<string,vector<double>> encodings;
unordered_map<string, long long int> encodingnum;
unordered_set<string> words;
vector<vector<double>> X_train;
vector<vector<double>> y_train;

//Define Windowsize
int windowsize = 3;

void CreateDataset(vector<string>* sentences, int size)
{
	fstream inp;
	inp.open("../Data/Sentences.csv", ios::in);

	cout << "Loading Dataset\n";
	string line;
	for(int i=0;i<size;i++)
	{
		try
		{
			getline(inp, line);
			if (!line.compare(""))
			{
				cout << "Empty Line Occured At " << i << " Position\n";
				break;
			}


			for (int i = 0; i < line.length(); i++)
			{
				if (line[i] == ',')
				{
					line = line.substr(i + 1);
					break;
				}
			}
			(*sentences).push_back(line);
		}
		catch (int i) { cout << "Error Occured On Line: " << i<<endl; break; }
	}

	inp.close();
	cout << "Dataset Loaded\n\n";
	cout << "Number of Sentences: " << sentences->size() << endl;
}

void OneHotEncoder()
{
	long long int counter = 0;
	long long int totalsize = words.size();

	cout << "\nBeginning Encoding" << endl;
	for(auto it=words.begin();it!=words.end();it++)
	{
		if (counter % 100 == 0)
			cout << "\r" << counter << "/" << totalsize;
		vector<double> encoding(totalsize);
		encoding[counter] = 1;
		string word = *it;

		encodingnum[word] = counter;
		encodings[word] = encoding;
		counter++;
	}

	cout << "\r" << totalsize << "/" << totalsize;
	cout << "\nEncoding Finished"<<endl;
}

vector<string> SentenceParser(string sentence)
{
	string word;
	vector<string> words;
	int wordstart = 0;

	for (int j = 0; j < sentence.length(); j++)
	{
		switch (sentence[j])
		{
		case ' ':
		case ',':
		case '.':
		case ';':
		case ':':
		case '~':
		case '!':
		case '$':
		case '^':
		case '&':
		case '(':
		case ')':
		case '*':
		case '_':
		case '-':
			word = sentence.substr(wordstart, j - wordstart);
			words.push_back(word);
			wordstart = j + 1;
			break;

		default:
			sentence[j] = tolower(sentence[j]);
			break;
		}
	}

	return words;
}

void BasicParser(vector<string>* sentences)
{
	cout << "\nBeginning Parsing" << endl;
	for (int i = 0; i < sentences->size(); i++)
	{
		if(i%100==0)
		cout << "\r" << i << "/" << sentences->size();

		string sentence = (*sentences)[i];
		string word;

		int wordstart = 0;
		for (int j = 0; j < sentence.length(); j++)
		{
			switch (sentence[j])
			{
			case ' ':
			case ',':
			case '.':
			case ';':
			case ':':
			case '~':
			case '!':
			case '$':
			case '^':
			case '&':
			case '(':
			case ')':
			case '*':
			case '_':
			case '-':
				word= sentence.substr(wordstart, j - wordstart);
				words.insert(word);
				wordstart = j+1;
				break;

			default:
				sentence[j] = tolower(sentence[j]);
				break;
			}
		}
	}
	cout << "\r" << sentences->size()<< "/" << sentences->size();
	cout << "\nParsing Finished" << endl;
}

void CreateTrainDataset(vector<string>* sentences, int windowsize)
{
	long long int totalsize = sentences->size();
	long long int encodingsize = encodings.begin()->second.size();

	cout << "\nCreating Train Dataset\n";
	for (int i = 0; i < totalsize-1; i++)
	{
		if(i%100==0)
		cout << "\r" << i << "/" << totalsize;

		string sentence = (*sentences)[i];
		vector<string> wordarray = SentenceParser(sentence);

		if (wordarray.size() < windowsize)
			continue;

		for (int j = 0; j <= wordarray.size() - windowsize; j++)
		{
			vector<double> input(encodingsize);
			for (int k = 0; k < windowsize; k++)
			{
				if (k != (int)(windowsize / 2))
					input[encodingnum[wordarray[j + k]]] = 1/(double)(windowsize);
			}
			X_train.push_back(input);
			y_train.push_back(encodings[wordarray[j + (int)(windowsize / 2)]]);
		}
	}
	cout << "\r" << totalsize << "/" << totalsize;
	cout << "\nTrain Dataset Created\n\n";
}

void WriteOutput(Network* model,string path)
{
	ofstream outfile(path);
	
	cout << "\nWriting To File" << endl;
	for (auto it = encodingnum.begin(); it != encodingnum.end(); it++)
	{
		try
		{
			string word = it->first;
			outfile << word << " (";
			for (int i = 0; i < (*model).layers[1].number; i++)
			{
				outfile << (*model).weights[0][it->second][i] << " ";
			}
			outfile << ")\n\n";
		}
		catch (int r)
		{}
	}
	cout << "Output Written\n";
	outfile.close();
}

int main()
{
	srand(time(NULL));

	vector<string> sentences;

	CreateDataset(&sentences, 500);
	BasicParser(&sentences);
	OneHotEncoder();
	words.clear();

	CreateTrainDataset(&sentences,windowsize);

	sentences.clear();
	encodings.clear();

	long long int onehotsize = y_train[0].size();
	long long int embeddingsize = 32;

	Network model;
	model.AddLayer(Layer(onehotsize, "Input"));
	model.AddLayer(Layer(embeddingsize, "Linear"));
	model.AddLayer(Layer(onehotsize, "Softmax"));

	model.lr = 1e-5;
	
	model.SetRegularizer("Elastic_Net");
	model.Regularizer.Elastic_Net_Alpha = 0.3;

	model.SetOptimizer("Adam");

	model.Compile("Stochastic");
	model.Summary();
	
	vector<vector<double>> predicted;

	model.Train(&X_train, &y_train,&predicted, 1, "MAE");
	
	X_train.clear();
	y_train.clear();

	WriteOutput(&model,"../Output/Embeddings32.txt");
	encodingnum.clear();
}
