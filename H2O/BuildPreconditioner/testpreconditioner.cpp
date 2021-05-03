#include <xerus.h>
//#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#define build_operator 0

using namespace xerus;
//using namespace Eigen;
using xerus::misc::operator<<;

//typedefs
//typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op(std::vector<value_t> coeffs);
TTOperator build_Fock_op_inv(std::vector<value_t>coeffs, size_t k, double shift);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
xerus::Tensor load_1e_int(std::string path);
xerus::Tensor load_2e_int(std::string path);


value_t get_hst(size_t k);
value_t get_tj(int j, size_t k);
value_t get_wj(int j, size_t k);
value_t minimal_ev(std::vector<value_t> coeffs);
value_t maximal_ev(std::vector<value_t> coeffs);


int main() {
	xerus::Index ii,jj,kk,ll,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4;
	size_t nob = 24;
//	std::string name = "hartreeFockEigenvvalues" + std::to_string(nob) +".csv";
//	Mat HFev_tmp = load_csv<Mat>(name);
	size_t rank = 1;
	size_t k = 700;
	double shift = 25;
	value_t hf_sol = -76.04055114 + 52.4190597253;

//	xerus::TTOperator op;
//	std::string name = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_benchmark.ttoperator";
//	read_from_disc(name,op);
//	op +=  shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
//	name = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
//	write_to_disc(name,op);


	std::vector<value_t> HFev;
//	for(size_t j = 0; j < nob; ++j){
//		auto val = HFev_tmp(j,0);
//		HFev.emplace_back(val);
//		HFev.emplace_back(val);
//	}
	Tensor T = load_1e_int("../data/T_H2O_48_bench.tensor");
	Tensor V = load_2e_int("../data/V_H2O_48_bench.tensor");
	for(size_t j = 0; j < 2*nob; ++j){
		value_t val = 0;
		//XERUS_LOG(info,"j = " << j);
		val +=T[{j,j}];
		//XERUS_LOG(info,"T_jj = " << val);
//		for (size_t k = 0; k < 2*nob; ++k)
//			val +=(V[{j,k,j,k}]-V[{j,k,k,j}]);
		for (size_t k : {0,1,2,3,22,23,30,31}){
		//	XERUS_LOG(info,"k = " << k);
		//	XERUS_LOG(info,V[{j,k,j,k}]);
		//	XERUS_LOG(info,V[{j,k,k,j}]);
					val +=(V[{j,k,j,k}]-V[{j,k,k,j}]);
		}
		XERUS_LOG(info,j << " value = " <<val);
		//XERUS_LOG(info,"T_jj  = " <<T[{j,j}]);
		HFev.emplace_back(val);
	}
	value_t test_hf = 0;
	for (size_t j : {0,1,2,3,22,23,30,31}){
		test_hf +=T[{j,j}];

			for (size_t k : {0,1,2,3,22,23,30,31})
				test_hf +=0.5*(V[{j,k,j,k}]-V[{j,k,k,j}]);
	}
	XERUS_LOG(info,"hf_sol = " << hf_sol);
	XERUS_LOG(info,"test_hf = " << test_hf);

	XERUS_LOG(info,"Build Fock OP");
	TTOperator Fock = build_Fock_op(HFev);
	Fock += shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
	TTOperator Fock_inv = build_Fock_op_inv(HFev, k, shift);
	std::string name2 = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	Fock_inv.round(0.0);

	write_to_disc(name2,Fock_inv);
	name2 = "../data/fock_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	write_to_disc(name2,Fock);



	XERUS_LOG(info,"Test");
	XERUS_LOG(info,"ranks Fock inv = " << Fock_inv.ranks());
	TTOperator test;

	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv(kk^(2*nob),jj^(2*nob));
	test += TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);

	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm()/TTOperator::identity(std::vector<size_t>(4*nob,2)).frob_norm());
	Fock_inv.round(1);
	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv(kk^(2*nob),jj^(2*nob));
	test += TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);

	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm()/TTOperator::identity(std::vector<size_t>(4*nob,2)).frob_norm());




//	XERUS_LOG(info, "Loading shifted Hamiltonian");
//	xerus::TTOperator Hs;
//  name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
//	read_from_disc(name2,Hs);
//	XERUS_LOG(info,"ranks Hamiltonian = " << Hs.ranks());
//	XERUS_LOG(info,"Norm Hamiltonian = " << Hs.frob_norm());
//
//	Tensor left1 = Tensor::ones({1,1,1,1});
//	Tensor left2 = Tensor::ones({1,1});
//	Tensor right1 = Tensor::ones({1,1,1,1});
//	Tensor right2 = Tensor::ones({1,1});
//	for (size_t i =0; i < 2*nob; ++i){
//		XERUS_LOG(info,i);
//		Tensor Fi = Fock_inv.get_component(i);
//		Tensor Hi = Hs.get_component(i);
//		left1(i1,i2,i3,i4) = left1(j1,j2,j3,j4)*Fi(j1,k1,k2,i1)*Hi(j2,k2,k3,i2)*Hi(j3,k3,k4,i3)*Fi(j4,k4,k1,i4);
//		left2(i1,i2) = left2(j1,j2)*Fi(j1,k1,k2,i1)*Hi(j2,k2,k1,i2);
//	}
//	right1() = left1(i1&0) * right1(i1&0);
//	right2() = left2(i1&0) * right2(i1&0);
//	value_t idnorm = (TTOperator::identity(std::vector<size_t>(4*nob,2))).frob_norm();
//
//	XERUS_LOG(info,"Approximation error = " << right1[0]);
//	XERUS_LOG(info,"Approximation error = " << right2[0]);
//	XERUS_LOG(info,"Approximation error = " << idnorm);
//	XERUS_LOG(info,"Approximation error = " << right1[0] +2*right2[0]+idnorm);
//


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


TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, const size_t k, double shift){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	TTOperator result(std::vector<size_t>(2*dim,2));
	int k_int = static_cast<int>(k);
	value_t coeff1;

	XERUS_LOG(info, "minimal = " << minimal_ev(coeffs));
	XERUS_LOG(info, "maximal = " << maximal_ev(coeffs));
	value_t lambda_min = maximal_ev(coeffs) + shift;

	for ( int j = -k_int; j <=k_int; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		for (size_t i = 0; i < dim; ++i){
			coeff1 = std::exp(2*get_tj(j,k)/lambda_min*(-coeffs[i]-shift/dim));
			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,1,1,0}] =  coeff1 ;
			aa[{0,0,0,0}] =  std::exp(2*get_tj(j,k)/lambda_min*(-shift/dim))  ;
			tmp.set_component(i,aa);
		}
		value_t coeff2 = 2*get_wj(j,k)/lambda_min;
		result -= coeff2 * tmp;
		//result.round(0.0);
		//XERUS_LOG(info,"j = " << j << " coeff2 " << coeff2 << " norm " << tmp.frob_norm()<< std::endl << result.ranks());
	}
	return result;
}
value_t get_hst(size_t k){
	return M_PI * M_PI / std::sqrt(static_cast<value_t>(k));
}

value_t get_tj(int j, size_t k){
	value_t hst = get_hst(k);
	return std::log(std::exp(static_cast<value_t>(j)*hst) + std::sqrt(1+std::exp(2*static_cast<value_t>(j)*hst)));
}

value_t get_wj(int j, size_t k){
	value_t hst = get_hst(k);
	return hst/std::sqrt(1+std::exp(-2*static_cast<value_t>(j)*hst));
}

value_t minimal_ev(std::vector<value_t> coeffs){
	value_t lambda;
	for (size_t i = 0; i < coeffs.size(); ++i){
		value_t coeff = coeffs[i];
		lambda += (coeff < 0 ? coeff : 0);
	}
	return lambda;
}

value_t maximal_ev(std::vector<value_t> coeffs){
	value_t lambda;
	for (size_t i = 0; i < coeffs.size(); ++i){
		value_t coeff = coeffs[i];
		lambda += (coeff < 0 ? 0 : coeff);
	}
	return lambda;
}

/*template<typename M>
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
*/
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


/*
 * Loads the 1 electron integral
 */
xerus::Tensor load_1e_int(std::string path){
	xerus::Tensor TT;
	std::ifstream read(path.c_str());
	misc::stream_reader(read,TT,xerus::misc::FileFormat::BINARY);
	return TT;
}

/*
 * Loads the 2 electron integral
 */
xerus::Tensor load_2e_int(std::string path){
	xerus::Tensor VV;
	std::ifstream read(path.c_str());
	misc::stream_reader(read,VV,xerus::misc::FileFormat::BINARY);
	return VV;
}
