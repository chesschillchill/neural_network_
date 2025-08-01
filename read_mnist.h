﻿/*
Chat GPT generated.
I don't understand the architecture of the MNIST dataset.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class ReadDataset
{
public:
    static int reverseInt(int i);
    // Function to read MNIST images
    static void read_mnist_images(const string &file_path, vector<vector<float>> &images);

    // Function to read MNIST labels
    static void read_mnist_labels(const string &file_path, vector<unsigned char> &labels);
};

// Function to reverse the byte order of an integer (big-endian to little-endian)
int ReadDataset::reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to read MNIST images
void ReadDataset::read_mnist_images(const string &file_path, vector<vector<float>> &images)
{
    ifstream file(file_path, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);

        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        images.resize(number_of_images, vector<float>(n_rows * n_cols));
        for (int i = 0; i < number_of_images; ++i)
        {
            vector<unsigned char> buffer(n_rows * n_cols);
            file.read((char*)buffer.data(), n_rows * n_cols);
            for (int j = 0; j < n_rows * n_cols; ++j)
                images[i][j] = static_cast<float>(buffer[j])/255.0f;
        }

    }
    else
    {
        cerr << "Unable to open file: " << file_path << endl;
    }
}

// Function to read MNIST labels
void ReadDataset::read_mnist_labels(const string &file_path, vector<unsigned char> &labels)
{
    ifstream file(file_path, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReadDataset::reverseInt(magic_number);

        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = ReadDataset::reverseInt(number_of_labels);

        labels.resize(number_of_labels);

        file.read((char *)labels.data(), number_of_labels);
    }
    else
    {
        cerr << "Unable to open file: " << file_path << endl;
    }
}
