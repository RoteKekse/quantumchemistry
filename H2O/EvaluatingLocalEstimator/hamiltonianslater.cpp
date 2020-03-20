#include <xerus.h>
#include <experimental/random>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace xerus;
using xerus::misc::operator<<;


class HamiltonianSlater{
	public:
		size_t d; // number pf site
		size_t p; //number of particles
		Tensor V;
		Tensor T;
		Tensor N;
		std::string path_T;
		std::string path_V;
		value_t nuc;


		HamiltonianSlater(size_t _d, size_t _p, std::string _path_T, std::string _path_V, value_t _nuc)
				: d(_d), p(_p),path_T(_path_T),path_V(_path_V), nuc(_nuc){
			T = load_1e_int();
			V = load_2e_int();
			N = load_nuclear();
		}

		void tester(std::vector<size_t> sample){
			value_t val = 0;
			for (size_t i : sample){
				val += T[{i,i}];
			}
			auto unit = make_unit_vec(sample);
			for (size_t i : sample){
				for (size_t j : sample){
					if (i != j){
//						auto tmpi= return_create(i,d);
//						auto tmpj = return_create(j,d);
//						auto tmpi2= return_annil(i,d);
//						auto tmpj2 = return_annil(j,d);
//						Tensor tmp;
//						Index i1,j1,k1,k2,k3;
//						XERUS_LOG(info,"-------------------");
//						tmp() = unit(i1&0) * tmpi(i1/2,k1/2)* tmpj(k1/2,k2/2)* tmpi2(k2/2,k3/2)* tmpj2(k3/2,j1/2) * unit(j1&0);
//						XERUS_LOG(info,"i = " << i << " j = " << j << " tmp = " << tmp);
//						tmp() = unit(i1&0) * tmpi(i1/2,k1/2)* tmpj(k1/2,k2/2)* tmpj2(k2/2,k3/2) * tmpi2(k3/2,j1/2) * unit(j1&0);
//						XERUS_LOG(info,"i = " << i << " j = " << j << " tmp = " << tmp);
						val += 0.5*(V[{i,j,i,j}] - V[{i,j,j,i}]);
//						XERUS_LOG(info,val);
					}
				}
			}
			XERUS_LOG(info,"The value for this unit vector should be: " << val);
		}

		TTTensor build(std::vector<size_t> sample){
			XERUS_LOG(HamiltonianSlater, "---- Building Operator ----");
			Index i1,j1,i2,j2,j3,j4;
			TTTensor result(std::vector<size_t>(d,2));
			TTOperator op(std::vector<size_t>(2*d,2));

			TTTensor tmp,tmp2,tmp3;
			auto unit = make_unit_vec(sample);

			//one electron operator
			for (size_t j : sample){
				auto site = build_1_site_operator(j ,0 ,0 , false);
				//tmp(i1/2,j1/2) = site(i1/2,j2/2) * tmp(j2/2,j1/2);
				tmp(i1&0) = site(i1/2,j1/2) * contract_j(j, unit, sample)(j1&0);
				result+=tmp;
			}
			result.round(0.0);

			//two electron operator
			for (size_t j = 0 ; j < d; ++j){ //TODO continue if j in sample
				XERUS_LOG(info,j);
				auto it = std::find (sample.begin(), sample.end(), j);
				TTTensor result_tmp(std::vector<size_t>(d,2));
				for (size_t l : sample){
					for (size_t k : sample){
						if  ((k == l) || (it != sample.end() && j != l && j != k))
							continue;
						auto site = build_1_site_operator(j ,k ,l , true);
						tmp(i1&0) = site(i1/2,j1/2) * contract_jkl(j, k, l, unit, sample)(j1&0);
						result_tmp+=0.5 * tmp;
					}
				}
				result += result_tmp;
				result.round(0.0);
			}
//			result(i1&0) = op(i1/2,j1/2) * unit(j1&0);
			return result;
		}

		TTTensor contract_jkl(size_t j, size_t k, size_t l, TTTensor unit, std::vector<size_t> sample){
			auto e1 = Tensor({1,2,1});
			auto e2 = Tensor({1,2,1});
			e1[0] = 1;
			e2[1] = 1;
			if (unit.component(k)[0] == 0 ){
				unit.set_component(k,e1);
				if (unit.component(l)[0] == 0){
					unit.set_component(l,e1);
					if (unit.component(j)[0] == 1){
						unit.set_component(j,e2);
						size_t sign = calculate_sign(j, k, l, sample);
						if (sign == 0)
							return unit;
						else
							return (-1)*unit;
					}
				}
			}
			return TTTensor(std::vector<size_t>(d,2));
		}

		size_t calculate_sign(size_t j, size_t k, size_t l, std::vector<size_t> sample){
			size_t kj =0,kk=0,kl=0;
			for (size_t i = 0 ; i < p; ++i){
				if (sample[i] < j) kj+=1;
				if (sample[i] < k) kk+=1;
				if (sample[i] < l) kl+=1;
			}
			if (k < l) kl-=1;
			if (k < j) kj-=1;
			if (l < j) kj-=1;
			return (kj+kk+kl) % 2;
		}

		TTTensor contract_j(size_t j, TTTensor unit, std::vector<size_t> sample){
			auto e1 = Tensor({1,2,1});
			e1[0] = 1;
			if ( unit.component(j)[0] == 0){
				unit.set_component(j,e1);
				size_t kj;
				for (size_t i = 0 ; i < p; ++i)
					if (sample[i] < j) kj+=1;
				size_t sign = (kj) % 2;
				if (sign == 0)
					return unit;
				else
					return (-1)*unit;
			}
			else
				return TTTensor(std::vector<size_t>(d,2));
		}

		std::vector<size_t> create_random_sample(){
			std::vector<size_t> res;
			size_t i = 0;
			while ( i < p){
				int r = std::experimental::randint(0, static_cast<int>(d-1));
				if(std::find(res.begin(), res.end(), r) == res.end()){
							res.emplace_back(r);
							++i;
				}
			}
			std::sort(res.begin(), res.end());
			return res;
		}

		TTTensor make_unit_vec(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				index[i] = 1;
			auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
			return unit;
		}





		TTOperator build_1_site_operator(size_t j, size_t k, size_t l, bool twoelec){
			TTOperator result(std::vector<size_t>(2*d,2));
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			auto s = xerus::Tensor::identity({2,2});
			s.reinterpret_dimensions({1,2,2,1});
			s[{0,1,1,0}] = -1.0;
			auto a = xerus::Tensor({1,2,2,1});
			a[{0,1,0,0}] = 1.0;
			for(size_t i = 0; i < d; ++i){
				value_t coeff = twoelec ? V[{i,j,k,l}]: T[{i,j}];
				if (i == 0){
					Tensor tmp = Tensor({1,2,2,2});
					tmp.offset_add(s,{0,0,0,0});
					tmp.offset_add(coeff*a,{0,0,0,1});
					result.set_component(i,tmp);
				} else if (i == d-1){
					Tensor tmp = Tensor({2,2,2,1});
					tmp.offset_add(coeff*a,{0,0,0,0});
					tmp.offset_add(id,{1,0,0,0});
					result.set_component(i,tmp);
				} else {
					Tensor tmp = Tensor({2,2,2,2});
					tmp.offset_add(s,{0,0,0,0});
					tmp.offset_add(coeff*a,{0,0,0,1});
					tmp.offset_add(id,{1,0,0,1});
					result.set_component(i,tmp);
				}
			}
			return result;
		}



		xerus::Tensor load_nuclear(){
			auto Nuc = xerus::Tensor({1});
			Nuc[0] = nuc;
			return Nuc;
		}

		/*
		 * Loads the 1 electron integral
		 */
		xerus::Tensor load_1e_int(){
			xerus::Tensor TT;
			std::ifstream read(path_T.c_str());
			misc::stream_reader(read,TT,xerus::misc::FileFormat::BINARY);
			return TT;
		}

		/*
		 * Loads the 2 electron integral
		 */
		xerus::Tensor load_2e_int(){
			xerus::Tensor VV;
			std::ifstream read(path_V.c_str());
			misc::stream_reader(read,VV,xerus::misc::FileFormat::BINARY);
			return VV;
		}

		/*
		 * Annihilation Operator with operator at position i
		 */
		xerus::TTOperator return_annil(size_t i, size_t d){ // TODO write tests for this
			xerus::Index i1,i2,jj, kk, ll;
			auto a_op = xerus::TTOperator(std::vector<size_t>(2*d,2));
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			auto s = xerus::Tensor::identity({2,2});
			s.reinterpret_dimensions({1,2,2,1});
			s[{0,1,1,0}] = -1.0;
			auto annhil = xerus::Tensor({1,2,2,1});
			annhil[{0,0,1,0}] = 1.0;
			for (size_t m = 0; m < d; ++m){
				auto tmp = m < i ? s : (m == i ? annhil : id );
				a_op.set_component(m, tmp);
			}
			return a_op;
		}

		/*
		 * Creation Operator with operator at position i
		 */
		xerus::TTOperator return_create(size_t i, size_t d){ // TODO write tests for this
			xerus::Index i1,i2,jj, kk, ll;
			auto c_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			auto s = xerus::Tensor::identity({2,2});
			s.reinterpret_dimensions({1,2,2,1});
			s[{0,1,1,0}] = -1.0;
			auto create = xerus::Tensor({1,2,2,1});
			create[{0,1,0,0}] = 1.0;
			for (size_t m = 0; m < d; ++m){
				auto tmp = m < i ? s : (m == i ? create : id );
				c_op.set_component(m, tmp);
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
};



int main(){
	XERUS_LOG(info, "---- Start building operator left to right! ----");
  time_t begin_time;

	size_t d = 50;
	size_t p = 10;
	std::string path_T = "../T_H2O_50.tensor";
	std::string path_V= "../V_H2O_50.tensor";
	value_t nuc = 8.80146457125193;
	HamiltonianSlater builder(d,p,path_T,path_V,nuc);
	auto sample = builder.create_random_sample();
	//std::vector<size_t> sample = {0,1,2,3,4,5,6,7,8,9};
	auto unit = builder.make_unit_vec(sample);
	builder.tester(sample);
	XERUS_LOG(info, "Loading Hamiltonian Operator");
	xerus::TTOperator H;
	std::string name = "../data/hamiltonian_H2O_" + std::to_string(d) +"_full_3.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,H,xerus::misc::FileFormat::BINARY);
	read.close();

	Index i1,i2,j1,j2;

	Tensor testH,testS, testE;



	begin_time = time (NULL);
	testH() = unit(i1&0)*H(i1/2,j1/2) * unit(j1&0);
	XERUS_LOG(info,"Time for contraction: " << time (NULL)-begin_time<<" sekunden");
	XERUS_LOG(info,"Norm testH " << testH);

	begin_time = time (NULL);
	auto test = builder.build(sample);
	testS() = test(i1&0) * unit(i1&0);
	XERUS_LOG(info,"Time for contraction: " << time (NULL)-begin_time<<" sekunden");
	XERUS_LOG(info,"Norm test " << testS);
	XERUS_LOG(info,"Norm diff " << (testS-testH).frob_norm());




	return 0;
}

