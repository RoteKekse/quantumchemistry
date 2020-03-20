#include <xerus.h>
#include <experimental/random>
//#include "ttcontainer.h"
#include <queue>
#include <omp.h>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

#include <memory>


#define debug 0
#define test 0
#define test2 0

using namespace xerus;
using xerus::misc::operator<<;

class PsiHScontract{
	public:
		const size_t d;
		const size_t particle;
		Tensor V;
		Tensor T;
		Tensor N;
		std::string path_T;
		std::string path_V;
		value_t nuc;

	private:
		std::vector<size_t> current_sample;
		std::vector<size_t> current_sample_k;
		std::vector<size_t> current_sample_inv;
		std::vector<size_t> idx;
		value_t result;

		int number_of_threads;

	public:
		/*
		 * Constructor
		 */
		PsiHScontract(size_t _d,size_t _p,std::string _path_T, std::string _path_V, value_t _nuc)
		: d(_d), particle(_p), path_T(_path_T),path_V(_path_V), nuc(_nuc), idx(d,0){
			T = load1eIntegrals();
			V = load2eIntegrals();
			N = loadNuclear();
			number_of_threads = 4;
		}

		void reset(){
			current_sample.clear();
			current_sample_k.clear();
			current_sample_inv.clear();
			idx = std::vector<size_t>(d,0);
			result = 0;
		}

		/*
		 * Build Operator
		 */
		value_t contract(const TTTensor psi,std::vector<size_t> sample){
			current_sample = sample;
			makeInvSampleAndIndex();
			Index i1,i2;
			result = 0;
			value_t signp = 1.0,signq = 1.0,signr =1.0,signs=1.0,val = 0;
			size_t nextp = 0,nextq = 0;

			// 1 e contraction
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
				idx[q] = 0; // annil operator a_q
				for (size_t p = 0; p < d; ++p){
					if (idx[p] == 1) {signp *= -1; continue;}
					val = T[{p,q}];
					if (std::abs(val) > 10e-12){
						idx[p] = 1; //creation
						result += signp *  val * psi[idx];
						idx[p] = 0; //annilation
					}
				}
				signq *= -1;
				signp = signq;
				idx[q] = 1; 				// creation operator a^*q
			}

			signq = 1.0; signr == 1.0; signs = 1.0; signp = 1.0;
			size_t count = 0;
			std::vector<value_t> coeffs;
			TTTensor tmp;
			Tensor tmp_res;

			bool nonzero = false;
			for (size_t r = 0; r < d; ++r){
				if (idx[r] != 1) continue;
				idx[r] = 0;
				signs = signr;
				for (size_t s = 0; s < r; ++s){
					if (idx[s] != 1) continue;
					idx[s] = 0;
					signq = signs;
					for (size_t q = 0; q < d; ++q){
						if (idx[q] == 1) {signq *= -1; continue;}
						//if (s%2 != q%2)  continue;
						idx[q] = 1;
						signp = signq;
						for (size_t p = 0; p < q; ++p){
							if (idx[p] == 1) { signp *= -1; coeffs.emplace_back(0); continue;}
							if ((p%2 != r%2 && q%2 != r%2) || (p%2 != s%2 && q%2 != s%2)) {coeffs.emplace_back(0); continue;}
							val = signp*(V[{p,q,r,s}] - V[{q,p,r,s}]);
							coeffs.emplace_back(val);
							if (std::abs(val) > 10e-12){
								nonzero = true;
//								if (q < 2){
									idx[p] = 1;
									result += val * psi[idx];
									idx[p] = 0;
//								}
							}
						}
//						if (q >= 2 and nonzero){
//							tmp = build1SiteTensor(coeffs, q);
//							tmp_res() = tmp(i1&0) * psi(i1&0);
//							result += tmp_res[0];
//						}
						nonzero = false;
						coeffs.clear();
						idx[q] = 0;
					}

					idx[s] = 1;
					signs *= -1;
				}
				idx[r] = 1;
				signr *= -1;
			}


			return result / psi[idx];
		}

		TTTensor build1SiteTensor(std::vector<value_t> coeffs, const size_t dim){
			XERUS_REQUIRE(dim >= 2,"Dimension too small");

			TTTensor result(std::vector<size_t>(d,2));
			value_t coeff = 0;
			for (size_t site = 0; site < d; ++site){
				bool active_s = idx[site] == 1 ? true : false;
				if ( site < dim){
					coeff = coeffs[site];
					if (site == 0){
						Tensor tmp({1,2,2});
						if (coeff != 0)
							tmp[{0,1,1}] = coeff;
						if (active_s)
							tmp[{0,1,0}] = -1;
						else
							tmp[{0,0,0}] = 1;
						result.set_component(site,tmp);
					} else if (site == dim-1){
						Tensor tmp({2,2,1});
						if (coeff != 0)
							tmp[{0,1,0}] = coeff;
						if (active_s)
							tmp[{1,1,0}] = 1;
						else
							tmp[{1,0,0}] = 1;
						result.set_component(site,tmp);
					}
					else {
						Tensor tmp({2,2,2});
						if (coeff != 0)
							tmp[{0,1,1}] = coeff;
						if (active_s){
							tmp[{0,1,0}] = -1;
							tmp[{1,1,1}] = 1;
						}	else {
							tmp[{0,0,0}] = 1;
							tmp[{1,0,1}] = 1;
						}
						result.set_component(site,tmp);
					}
				} else {
					if (active_s)
						result.component(site)[{0,1,0}] = 1;
					else
						result.component(site)[{0,0,0}] = 1;
				}
			}
			return result;
		}

		void makeInvSampleAndIndex(){
			for (size_t i = 0; i < d; ++i){
				if(!std::binary_search (current_sample.begin(), current_sample.end(), i))
					current_sample_inv.emplace_back(i);
				else
					idx[i] = 1;
			}
		}


		/*
		 * Loads the nuclear Potential
		 */
		xerus::Tensor loadNuclear(){
				auto Nuc = xerus::Tensor({1});
				Nuc[0] = nuc;
				return Nuc;
			}

		/*
		 * Loads the 1 electron integral
		 */
		xerus::Tensor load1eIntegrals(){
			xerus::Tensor TT;
			std::ifstream read(path_T.c_str());
			misc::stream_reader(read,TT,xerus::misc::FileFormat::BINARY);
			return TT;
		}

		/*
		 * Loads the 2 electron integral
		 */
		xerus::Tensor load2eIntegrals(){
			xerus::Tensor VV;
			std::ifstream read(path_V.c_str());
			misc::stream_reader(read,VV,xerus::misc::FileFormat::BINARY);
			return VV;
		}


		std::vector<size_t> createRandomSample(){
			std::vector<size_t> res;
			size_t i = 0;
			while ( i < particle){
				int r = std::experimental::randint(0, static_cast<int>(d-1));
				if(std::find(res.begin(), res.end(), r) == res.end()){
							res.emplace_back(r);
							++i;
				}
			}
			std::sort(res.begin(), res.end());
			return res;
		}

		TTTensor makeUnitVector(std::vector<size_t> sample){
			return makeUnitVector(sample,d);
		}


		TTTensor makeUnitVector(std::vector<size_t> sample,size_t dim){
			std::vector<size_t> index(dim, 0);
			for (size_t i : sample)
				if (i < dim)
					index[i] = 1;
			auto unit = TTTensor::dirac(std::vector<size_t>(dim,2),index);
			return unit;
		}

		TTOperator build_brute(const size_t dim){
			Index ii,jj;
			auto opH = xerus::TTOperator(std::vector<size_t>(2*dim,2));
			for (size_t i = 0; i < dim; i++){
				XERUS_LOG(info,i);
					for (size_t j = 0; j < dim; j++){
							value_t val = T[{i , j}];
							auto one_e_ac = return_one_e_ac(i,j,dim);
							opH += val * one_e_ac;
					}
			}

			auto opV = xerus::TTOperator(std::vector<size_t>(2*dim,2));
//			for (size_t i = 0; i < dim; i++){
//				for (size_t j = 0; j < dim; j++){
//					for (size_t k = 0; k < dim; k++){
//						for (size_t l = 0; l < dim; l++){
//							value_t val = V[{i,j,k,l}];
//
//							auto two_e_ac = return_two_e_ac(i,j,l,k,dim);
//							if (std::abs(val) < 10e-14 || i == j || k == l )//|| a == b || c: == d) // TODO check i == j || k == l
//									continue;
//							opV += 0.5*val * two_e_ac; // TODO: Note change back swap k,l
//						}
//					}
//				}
//			}
			auto res = opH + opV;
			res.round(0.0);
			return res;
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
		xerus::TTOperator return_two_e_ac_partial3(size_t i, size_t j, size_t d){ //todo test
				auto cr1 = return_create(i,d);
				auto cr2 = return_create(j,d);
				xerus::TTOperator res;
				xerus::Index ii,jj,kk,ll;
				res(ii/2,kk/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) ;
				return res;
			}
		xerus::TTOperator return_two_e_ac_partial4(size_t i, size_t j,  size_t d){ //todo test
			auto an1 = return_annil(i,d);
			auto an2 = return_annil(j,d);
				xerus::TTOperator res;
				xerus::Index ii,jj,kk,ll;
				res(ii/2,kk/2) = an1(ii/2,jj/2) * an2(jj/2,kk/2)  ;
				return res;
			}
		xerus::TTOperator return_two_e_ac_partial(size_t i, size_t j, size_t k, size_t d){ //todo test
			auto cr1 = return_create(i,d);
			auto cr2 = return_create(j,d);
			auto an1 = return_annil(k,d);
			xerus::TTOperator res;
			xerus::Index ii,jj,kk,ll;
			res(ii/2,ll/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) ;
			return res;
		}
		xerus::TTOperator return_two_e_ac_partial2(size_t i, size_t j, size_t k, size_t d){ //todo test
			auto cr1 = return_create(i,d);
			auto an1 = return_annil(j,d);
			auto an2 = return_annil(k,d);
			xerus::TTOperator res;
			xerus::Index ii,jj,kk,ll;
			res(ii/2,ll/2) = cr1(ii/2,jj/2) * an1(jj/2,kk/2) * an2(kk/2,ll/2) ;
			return res;
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



};


