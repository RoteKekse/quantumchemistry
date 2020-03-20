#include <xerus.h>
using namespace xerus;
using xerus::misc::operator<<;

#pragma once

void write_to_disc(std::string name, TTOperator &op){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
	write.close();
}

void write_to_disc(std::string name, TTTensor &x){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,x,xerus::misc::FileFormat::BINARY);
	write.close();
}

void write_to_disc(std::string name, Tensor &x){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,x,xerus::misc::FileFormat::BINARY);
	write.close();
}


void read_from_disc(std::string name, TTOperator &op){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();
}

void read_from_disc(std::string name, TTTensor &x){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,x,xerus::misc::FileFormat::BINARY);
	read.close();
}

void read_from_disc(std::string name, Tensor &x){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,x,xerus::misc::FileFormat::BINARY);
	read.close();
}


