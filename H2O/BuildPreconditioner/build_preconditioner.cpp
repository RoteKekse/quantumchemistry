#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#define build_operator 1

using namespace xerus;
using namespace Eigen;
using xerus::misc::operator<<;

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op(std::vector<value_t> coeffs);
TTOperator build_Fock_op_inv(std::vector<value_t>coeffs, size_t k);
TTOperator build_pseudo_id(size_t dim);
value_t minimal_ev(std::vector<value_t> coeffs);
xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);


value_t get_hst(size_t k);
value_t get_tj(int j, size_t k);
value_t get_wj(int j, size_t k);




int main() {
	xerus::Index ii,jj,kk,ll;
	size_t nob = 25;
	std::string name = "hartreeFockEigenvvalues" + std::to_string(nob) +".csv";
	Mat HFev_tmp = load_csv<Mat>(name);




	std::vector<value_t> HFev;
	for(size_t j = 0; j < nob; ++j){
		auto val = HFev_tmp(j,0) < 0 ? HFev_tmp(j,0) : -HFev_tmp(j,0);
		HFev.emplace_back(val);
		HFev.emplace_back(val);
	}

	Tensor T_MO;
	name = "../T_H2O_50.tensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,T_MO,xerus::misc::FileFormat::BINARY);
	read.close();

	Tensor V_MO;
	name = "../V_H2O_50.tensor";
	std::ifstream read2(name.c_str());
	misc::stream_reader(read2,V_MO,xerus::misc::FileFormat::BINARY);
	read2.close();
#if build_operator
	double sev =0;
	XERUS_LOG(info, "Build 1e");
	auto opH = xerus::TTOperator(std::vector<size_t>(4*nob,2));
	for (size_t i = 0; i < 2*nob; i++){
			value_t val = T_MO[{i , i}];
			sev += val < 0 ? val : 0;
			opH += val * return_one_e_ac(i,i,2*nob);
			opH.round(0.0);
	}
	XERUS_LOG(info,"sev = " << sev);
	XERUS_LOG(info, "Build 2e");
	auto opV = xerus::TTOperator(std::vector<size_t>(4*nob,2));
	for (size_t i = 0; i < 2*nob; i++){
		XERUS_LOG(info, i);
		for (size_t j = 0; j < 2*nob; j++){
			if(j!=i){
				value_t val = V_MO[{i,j,i,j}];
				value_t val2 = V_MO[{i,j,j,i}];
				sev += val < 0 ? val : 0;
				sev += val2 < 0 ? val2 : 0;
				opV += 0.5*val * return_two_e_ac(i,j,j,i,2*nob);
				opV += 0.5*val2 * return_two_e_ac(i,j,i,j,2*nob);
				opV.round(0.0);
			}
		}
	}
	XERUS_LOG(info,"sev = " << sev);

	XERUS_LOG(info,"Norm opH" << opH.frob_norm());
	XERUS_LOG(info,"Norm opV" << opV.frob_norm());
	xerus::TTOperator op = opH + opV;
	op.round(0.0);

  std::string name2 = "../data/FockOperator_H2O" + std::to_string(2*nob) +"_full.ttoperator";
	std::ofstream write(name2.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
	write.close();
#else
	xerus::TTOperator op;
	//name = "../FockOperator_H2O" + std::to_string(2*nob) +"_full.ttoperator";
	name = "../data/hamiltonian_H2O_50_full_2.ttoperator";
	std::ifstream read3(name.c_str());
	misc::stream_reader(read3,op,xerus::misc::FileFormat::BINARY);
	read3.close();
#endif

	XERUS_LOG(inf, "Ranks of Fock operator: " << op.ranks());
	XERUS_LOG(info, HFev);



	TTTensor vec(std::vector<size_t>(2*nob,2));
	Tensor unitvec1 = Tensor::dirac({1,2,1},{0,0,0});
	Tensor unitvec2 = Tensor::dirac({1,2,1},{0,1,0});
	for (size_t i = 0; i < 2*nob; ++i)
		vec.set_component(i,unitvec1);
	TTOperator test;
	TTTensor test2;
	Tensor test3;

	test3() = op(ii/2,jj/2) * vec(jj&0) * vec(ii&0);
	XERUS_LOG(info, "Test with vac = " << test3[0]);

//
//	for (size_t i = 0; i < 2*nob; ++i){
//		TTTensor tmp(std::vector<size_t>(2*nob,2));
//		for (size_t j = 0; j < 2*nob; ++j)
//			tmp.set_component(j,(i==j ? unitvec2 :unitvec1));
//		test3() = op(ii/2,jj/2) * tmp(jj&0) * tmp(ii&0);
//		XERUS_LOG(info, "Test with 1e = " << test3[0]);
//	}
//
//	{
//		TTTensor tmp(std::vector<size_t>(2*nob,2));
//		for (size_t j = 0; j < 2*nob; ++j)
//			tmp.set_component(j,(j < 10 ? unitvec2 :unitvec1));
//		test3() = op(ii/2,jj/2) * tmp(jj&0) * tmp(ii&0);
//		XERUS_LOG(info, "Test with 10e = " << test3[0]);
//	}
//
//


	for(size_t i=0; i < 2*nob; ++i){
		value_t tmp = 0;
		for(size_t j = 0; j < 2*nob; ++j){
			if (i!=j)
				tmp+= 0.5*(V_MO[{i,j,i,j}] - V_MO[{i,j,j,i}]);
		}
//		XERUS_LOG(info, T_MO[{i,i}]);
//		XERUS_LOG(info, tmp);
//		XERUS_LOG(info, T_MO[{i,i}]+tmp);
//		XERUS_LOG(info, T_MO[{i,i}]-tmp);
	}



	std::vector<value_t> test_Fock;
	test_Fock.emplace_back(-1);
	test_Fock.emplace_back(-2);
	test_Fock.emplace_back(-3);

	XERUS_LOG(info,"Build Fock OP");
	TTOperator Fock = build_Fock_op(test_Fock);

	TTOperator Focksquare;
	Focksquare(ii/2,jj/2) = Fock(ii/2,kk/2)*Fock(kk/2,jj/2);
	XERUS_LOG(info,Focksquare.ranks());
	Focksquare.round(0.0);
	XERUS_LOG(info,Focksquare.ranks());
	for(size_t i = 0; i < Focksquare.order()/2;i++){
		XERUS_LOG(info, Fock.get_component(i));
		XERUS_LOG(info, Focksquare.get_component(i));
	}

	Tensor FockT(Fock);
	FockT.reinterpret_dimensions({8,8});
	XERUS_LOG(info,FockT);


	XERUS_LOG(info,"Build Fock OP inverse");
	TTOperator Fock_inv = build_Fock_op_inv(test_Fock,100);
	XERUS_LOG(info,"Build Pseudo inverse");
	TTOperator pseudo_id = build_pseudo_id(test_Fock.size());

	Tensor FockT_inv(Fock_inv);
	FockT_inv.reinterpret_dimensions({8,8});
	XERUS_LOG(info,FockT_inv);
	XERUS_LOG(info,Fock_inv.ranks());





	Fock.require_correct_format();
	Fock_inv.require_correct_format();
	pseudo_id.require_correct_format();


	test(ii/2,jj/2) = Fock(ii/2,kk/2) * Fock_inv(kk/2,ll/2) * pseudo_id(ll/2,jj/2);
	XERUS_LOG(info, "test2 norm = " << (test - pseudo_id).frob_norm());///pseudo_id.frob_norm());



//	XERUS_LOG(info, "test error = " << (TTOperator::identity(std::vector<size_t>(4*nob,2)) - test).frob_norm());


	return 0;
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

TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, const size_t k){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	value_t lambda_min = -1.0 * minimal_ev(coeffs);
	TTOperator pseudo_id = build_pseudo_id(dim);
	int k_int = static_cast<int>(k);
	TTOperator inv(std::vector<size_t>(2*dim,2));
	for ( int j = -k_int; j <=k_int; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		//XERUS_LOG(info,"j = " << j << ": " << get_tj(j,k));
		for (size_t i = 0; i < dim; ++i){
			value_t coeff1 = std::exp(2*get_tj(j,k)/lambda_min*coeffs[i]);
			//XERUS_LOG(info,"coeff1 " << coeff1);

			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,1,1,0}] = coeff1;
			aa[{0,0,0,0}] = 1;
			tmp.set_component(i,aa);
		}
		//XERUS_LOG(info,"j = " << j << " k = " << k << " wjk = " << get_wj(j,k));
		//XERUS_LOG(info,"j = " << j << " k = " << k << " tjk = " << get_tj(j,k));

		value_t coeff2 = 2*get_wj(j,k)/lambda_min;
		//XERUS_LOG(info,"coeff2 " << coeff2);
		//XERUS_LOG(info,"lambda_min " << lambda_min);

		//XERUS_LOG(info,tmp.frob_norm());
		inv -= coeff2 * tmp;
		//project solution
		inv(ii/2,jj/2) = inv(ii/2,kk/2) * pseudo_id(kk/2,jj/2);
		inv.round(0.0);
	}
	return inv;
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

TTOperator build_pseudo_id(size_t dim){
	TTOperator result(std::vector<size_t>(2*dim,2));

	auto id = xerus::Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto aa = xerus::Tensor({1,2,2,1});
	aa[{0,0,0,0}] = 1.0;
	for (size_t comp = 0; comp < dim; ++comp){
		if (comp == 0){
				Tensor tmp = Tensor({1,2,2,2});
				tmp.offset_add(id,{0,0,0,0});
				tmp.offset_add((-1.0)*aa,{0,0,0,1});
				result.set_component(comp,tmp);
		} else if (comp == dim - 1){
			Tensor tmp = Tensor({2,2,2,1});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(aa,{1,0,0,0});
			result.set_component(comp,tmp);
		} else {
			Tensor tmp = Tensor({2,2,2,2});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(aa,{1,0,0,1});
			result.set_component(comp,tmp);
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
