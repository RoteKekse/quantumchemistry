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

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_diag_op(size_t d);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);

xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);
xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);





int main() {
	xerus::Index ii,jj,kk,ll,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4;
	size_t nob = 25;
	size_t rank = 1;
	size_t k = 700;

	double shift = 86.0;


	XERUS_LOG(info,"Build Diagonal");
	TTOperator diag = build_diag_op(2*nob);
	diag += shift*TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info,"Load Hamiltonian");
	xerus::TTOperator H,F;
  std::string name = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,H);

	name = "../data/fock_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,F);

	std::vector<size_t> idx(2*nob,0);
	idx[0] = 1;
	idx[1] = 1;
	idx[2] = 1;
	idx[3] = 1;
	idx[4] = 1;
	idx[5] = 1;
	idx[6] = 1;
	idx[7] = 1;
	idx[8] = 1;
	idx[9] = 1;
	//idx[10] = 1;
	//idx[11] = 1;
	//idx[12] = 1;
	//idx[40] = 1;
	idx.insert(idx.end(),idx.begin(),idx.end());
	XERUS_LOG(info,std::setprecision(8) << H[idx] << " " << diag[idx] << " " << F[idx] << " "<< std::abs(H[idx] - diag[idx]));
	diag.round(10e-12);
	XERUS_LOG(info,diag.ranks());

//	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm()/TTOperator::identity(std::vector<size_t>(4*nob,2)).frob_norm());

	TTOperator n13I = return_two_e_ac(0,2,2,0,4) + TTOperator::identity(std::vector<size_t>(2*4,2));
	XERUS_LOG(info,n13I.ranks());
	n13I.round(10e-15);
	XERUS_LOG(info,n13I.ranks());

	TTOperator n1I = return_one_e_ac(0,0,4) + TTOperator::identity(std::vector<size_t>(2*4,2));
	XERUS_LOG(info,n1I.ranks());
	n1I.round(10e-15);
	XERUS_LOG(info,n1I.ranks());

	return 0;
}

/*
 * Loads the 1 electron integral
 */
xerus::Tensor load_1e_int(){
	xerus::Tensor TT;
	std::ifstream read("../T_H2O_50.tensor");
	misc::stream_reader(read,TT,xerus::misc::FileFormat::BINARY);
	return TT;
}

/*
 * Loads the 2 electron integral
 */
xerus::Tensor load_2e_int(){
	xerus::Tensor VV;
	std::ifstream read("../V_H2O_50.tensor");
	misc::stream_reader(read,VV,xerus::misc::FileFormat::BINARY);
	return VV;
}


TTOperator build_diag_op(size_t d){
	TTOperator result(std::vector<size_t>(2*d,2));
	Tensor T,V;
	T = load_1e_int();
	V = load_2e_int();

	for (size_t i = 0; i < d; ++i){
		result += T[{i,i}] * return_one_e_ac(i, i, d);
	}

	for (size_t i = 0; i < d; ++i){
		for (size_t j = 0; j < d; ++j){
			result+= 0.5 * (V[{i,j,i,j}] - V[{i,j,j,i}]) * return_two_e_ac(i, j,j,i, d);
		}
	}
	return result;
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
