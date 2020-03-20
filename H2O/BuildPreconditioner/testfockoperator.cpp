#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#define build_operator 0

using namespace xerus;
using namespace Eigen;
using xerus::misc::operator<<;
xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);
xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op(std::vector<value_t> coeffs);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);



xerus::Tensor load_1e_int(){
	xerus::Tensor TT;
	std::string path_T = "../T_H2O_50.tensor";
	std::ifstream read(path_T.c_str());
	misc::stream_reader(read,TT,xerus::misc::FileFormat::BINARY);
	return TT;
}
xerus::Tensor load_2e_int(){
	std::string path_V= "../V_H2O_50.tensor";
	xerus::Tensor VV;
	std::ifstream read(path_V.c_str());
	misc::stream_reader(read,VV,xerus::misc::FileFormat::BINARY);
	return VV;
}



int main() {
	xerus::Index ii,jj,kk,ll,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4;
	size_t nob = 25;
	std::string name = "hartreeFockEigenvvalues" + std::to_string(nob) +".csv";
	Mat HFev_tmp = load_csv<Mat>(name);
	size_t rank = 1;
	size_t k = 280;
	xerus::Tensor T = load_1e_int();
	xerus::Tensor V = load_2e_int();


	std::vector<value_t> HFev;
	for(size_t j = 0; j < nob; ++j){
		auto val = HFev_tmp(j,0);
		HFev.emplace_back(val);
		HFev.emplace_back(val);
	}
	XERUS_LOG(info,HFev);

	XERUS_LOG(info,"Build Fock OP");
	TTOperator Fock = build_Fock_op(HFev);

	TTOperator Fock2(std::vector<size_t>(4*nob,2));
	for (size_t i = 0; i < 2*nob; ++i){
		Fock2 += T[{i,i}] * return_one_e_ac(i,i,2*nob);
		for (size_t j = 0; j < 10; ++j){
			Fock2 += (V[{i,j,i,j}] - V[{i,j,j,i}]) * return_one_e_ac(i,i,2*nob);
		}
	}
	XERUS_LOG(info,"Fock  = " << (Fock).frob_norm());
	XERUS_LOG(info,"Fock2 = " << (Fock2).frob_norm());
	XERUS_LOG(info,"Diff  = " << (Fock-Fock2).frob_norm());


	return 0;
}

TTOperator build_Fock_op(std::vector<value_t> coeffs){
	size_t dim = coeffs.size();

	TTOperator result(std::vector<size_t>(2*dim,2));
	size_t comp = 0;
	auto id = xerus::Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto aa = xerus::Tensor({1,2,2,1});
	aa[{0,1,1,0}] = 1.0;
	for (size_t comp = 0; comp < dim; ++comp){
		value_t coeff = coeffs[comp];
		if (comp == 0){
				Tensor tmp = Tensor({1,2,2,2});
				tmp.offset_add(id,{0,0,0,0});
				tmp.offset_add(coeff*aa,{0,0,0,1});
				result.set_component(comp,tmp);
		} else if (comp == dim - 1){
			Tensor tmp = Tensor({2,2,2,1});
			tmp.offset_add(coeff*aa,{0,0,0,0});
			tmp.offset_add(id,{1,0,0,0});
			result.set_component(comp,tmp);
		} else {
			Tensor tmp = Tensor({2,2,2,2});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(coeff*aa,{0,0,0,1});
			tmp.offset_add(id,{1,0,0,1});
			result.set_component(comp,tmp);
		}
	}
	return result;
}



template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

void write_to_disc(std::string name, TTOperator &op){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
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


//Creation of Operators
xerus::TTOperator return_annil(size_t i, size_t d){ // TODO write tests for this


	xerus::Index i1,i2,jj, kk, ll;
	auto a_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto annhil = xerus::Tensor({2,2});
	annhil[{0,1}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? s : (m == i ? annhil : id );
		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];
		a_op.set_component(m, res);
	}
	return a_op;
}

xerus::TTOperator return_create(size_t i, size_t d){ // TODO write tests for this
	xerus::Index i1,i2,jj, kk, ll;
	auto c_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto create = xerus::Tensor({2,2});
	create[{1,0}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? s : (m == i ? create : id );
		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];
		c_op.set_component(m, res);
	}
	return c_op;
}

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d){ // TODO write tests for this
	auto cr = return_create(i,d);
	auto an = return_annil(j,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk;
	res(ii/2,jj/2) = cr(ii/2,kk/2) * an(kk/2, jj/2);
	return res;
}

xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d){ //todo test
	auto cr1 = return_create(i,d);
	auto cr2 = return_create(j,d);
	auto an1 = return_annil(k,d);
	auto an2 = return_annil(l,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk,ll,mm;
	res(ii/2,mm/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) * an2(ll/2,mm/2);
	return res;
}
