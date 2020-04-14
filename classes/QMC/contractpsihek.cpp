#include <xerus.h>
#include <experimental/random>
//#include "ttcontainer.h"
#include <queue>
#include <omp.h>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <unordered_map>

#include "../../classes/containerhash.cpp"
#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

#include <memory>

#pragma once

#define debug 0
#define test 0
#define test2 0

using namespace xerus;
using xerus::misc::operator<<;

class ContractPsiHek{
	public:
		const size_t d;
		const size_t p;
		Tensor V;
//		Tensor V2;
		Tensor T;
//		Tensor T2;
		Tensor N;
		std::string path_T;
		std::string path_V;
		value_t nuc;
		value_t shift;
		TTTensor psi;

	private:
		std::vector<size_t> current_sample;
		std::vector<size_t> current_sample_k;
		std::vector<size_t> current_sample_inv;
		std::vector<size_t> idx;
		value_t result;
	public:
		std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> umap_psi; // storing evaluations of psi
		std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> umap2_psi; // storing evaluations of psi

	public:
		/*
		 * Constructor
		 */
		ContractPsiHek(TTTensor _psi,size_t _d,size_t _p,std::string _path_T, std::string _path_V, value_t _nuc, value_t _shift)
		: psi(_psi), d(_d), p(_p), path_T(_path_T),path_V(_path_V), nuc(_nuc), idx(d,0), shift(_shift){
			T = load1eIntegrals();
			V = load2eIntegrals();
			N = loadNuclear();
//			read_from_disc("../data/T_H2O_48_bench.tensor",T2);
//			read_from_disc("../data/V_H2O_48_bench.tensor",V2);
			XERUS_LOG(info, "T sparse? " << T.is_sparse());
			XERUS_LOG(info, "V sparse? " << V.is_sparse());
		}

		ContractPsiHek( const ContractPsiHek&  _other ) = default;


		void reset(std::vector<size_t> sample){
			current_sample.clear();
			current_sample_k.clear();
			current_sample_inv.clear();
			idx = std::vector<size_t>(d,0);
			result = 0;
			current_sample = sample;
			makeInvSampleAndIndex();
		}

		void reset_psi(TTTensor _psi){
			umap_psi.clear();
			psi = _psi;
		}


		/*
		 * Contraction, does psi H ek for current sample, NOTE: use reset first
		 */
		value_t contract(){
			result = 0;
			value_t signp = 1.0,signq = 1.0,signr =1.0,signs=1.0,val = 0,val1=0,val2 = 0;
			size_t nextp = 0,nextq = 0;

			// 1 e contraction
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
				idx[q] = 0; // annil operator a_q
				for (size_t p = 0; p < d; ++p){
					if (idx[p] == 1) {signp *= -1; continue;}
					if (p%2 != q%2) {continue;}
					val = returnTValue(p/2,q/2);
//					if (std::abs(val - T2[{p,q}]) > 1e-14)
//						XERUS_LOG(info,p << " " << q << " " << val);
					if (std::abs(val) > 10e-12){
						idx[p] = 1; //creation
						auto itr = umap_psi.find(idx);
						if (itr == umap_psi.end())
							umap_psi[idx] = psi[idx];
						result += signp *  val * umap_psi[idx];
						idx[p] = 0; //annilation
					}
				}
				signq *= -1;
				signp = signq;
				idx[q] = 1; 				// creation operator a^*q
			}

			signq = 1.0; signr = 1.0; signs = 1.0; signp = 1.0;
			size_t count1 = 0, count2 = 0,count3 = 0;
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
						idx[q] = 1;
						signp = signq;
						for (size_t p = 0; p < q; ++p){
							if (idx[p] == 1) { signp *= -1;  continue;}
							count1++;
							if (not ((p%2 == r%2 and q%2 == s%2) or (p%2 == s%2 and q%2 == r%2))) {continue;}
							count2++;
							val1 = ((p%2 != r%2) || (q%2!=s%2)) ? 0 : returnVValue(p/2,q/2,r/2,s/2);
							val2 = ((p%2 != s%2) || (q%2!=r%2)) ? 0 : returnVValue(q/2,p/2,r/2,s/2);
							val = signp*(val1 - val2);
//							if (std::abs(val - signp*(V2[{p,q,r,s}]-V2[{q,p,r,s}])) > 1e-14)
//															XERUS_LOG(info,p << " " << q << " " << r << " " << s << " " << val << " " << signp*(V2[{p,q,r,s}]-V2[{q,p,r,s}]));

							if (std::abs(val) > 10e-8){
								idx[p] = 1;
								auto itr = umap_psi.find(idx);
								if (itr == umap_psi.end())
									umap_psi[idx] = psi[idx];
								count3++;
								result += val * umap_psi[idx];
								idx[p] = 0;
							}
						}
						idx[q] = 0;
					}
					idx[s] = 1;
					signs *= -1;
				}
				idx[r] = 1;
				signr *= -1;
			}
//			XERUS_LOG(info,count1);
//			XERUS_LOG(info,count2);
//			XERUS_LOG(info,count3);
			return result + shift * psiEntry();
		}

		value_t contract2(){
			result = 0;
			value_t signp = 1.0,signq = 1.0,signr =1.0,signs=1.0,val = 0,val1=0,val2 = 0;
			size_t nextp = 0,nextq = 0;
			XERUS_LOG(info,"Start index\n" << idx);

			// 1 e contraction
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
				idx[q] = 0; // annil operator a_q
				for (size_t p = 0; p < d; ++p){
					if (idx[p] == 1) {signp *= -1; continue;}
					if (p%2 != q%2) {continue;}
					val = returnTValue(p/2,q/2);
//					if (std::abs(val - T2[{p,q}]) > 1e-14)
//						XERUS_LOG(info,p << " " << q << " " << val);
					if (std::abs(val) > 10e-8){
						idx[p] = 1; //creation
						auto itr = umap2_psi.find(idx);
						if (itr == umap2_psi.end())
							XERUS_LOG(info,"Not Found\n" << idx);
						result += signp *  val * umap2_psi[idx];
						idx[p] = 0; //annilation
					}
				}
				signq *= -1;
				signp = signq;
				idx[q] = 1; 				// creation operator a^*q
			}

			signq = 1.0; signr = 1.0; signs = 1.0; signp = 1.0;
			size_t count1 = 0, count2 = 0,count3 = 0;
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
						idx[q] = 1;
						signp = signq;
						for (size_t p = 0; p < q; ++p){
							if (idx[p] == 1) { signp *= -1;  continue;}
							count1++;
							if (not ((p%2 == r%2 and q%2 == s%2) or (p%2 == s%2 and q%2 == r%2))) {continue;}
							count2++;
							val1 = ((p%2 != r%2) || (q%2!=s%2)) ? 0 : returnVValue(p/2,q/2,r/2,s/2);
							val2 = ((p%2 != s%2) || (q%2!=r%2)) ? 0 : returnVValue(q/2,p/2,r/2,s/2);
							val = signp*(val1 - val2);
//							if (std::abs(val - signp*(V2[{p,q,r,s}]-V2[{q,p,r,s}])) > 1e-14)
//															XERUS_LOG(info,p << " " << q << " " << r << " " << s << " " << val << " " << signp*(V2[{p,q,r,s}]-V2[{q,p,r,s}]));

							if (std::abs(val) > 10e-8){
								idx[p] = 1;
								auto itr = umap2_psi.find(idx);
								if (itr == umap2_psi.end())
									XERUS_LOG(info,"Not Found\n" << idx);
								count3++;
								result += val * umap2_psi[idx];
								idx[p] = 0;
							}
						}
						idx[q] = 0;
					}
					idx[s] = 1;
					signs *= -1;
				}
				idx[r] = 1;
				signr *= -1;
			}
//			XERUS_LOG(info,count1);
//			XERUS_LOG(info,count2);
//			XERUS_LOG(info,count3);
			return result + shift * psiEntry();
		}


		void preparePsiEval(){ 			// TODO can one keep the lower contractions for different e_ks??
			Index r1,r2,r3;
			// a queue containing a pair of the position and a vector containing of the data pairs index and Tensor
			// the index of the data vector is the linearized version of an order 4 Tensor for the number of
			// annihilated (max. 2), created (max 2.), spin up (max. p//2) and, spin down (max. p//2) particles
			std::queue<std::pair<size_t,std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>>>> queue;
			std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>> data_tmpl;
			for (size_t i = 0; i < 3*3*(p/2+1)*(p/2+1); ++i){
				std::vector<std::pair<std::vector<size_t>,Tensor>> tmp;
				data_tmpl.emplace_back(tmp);
			}

			// initialize queue with slices of TT Tensor
			for (size_t i = 0; i < d; ++i){
				auto psi0 = psi.get_component(i);
				auto psi1 = psi.get_component(i);
				psi0.fix_mode(1,0);
				psi1.fix_mode(1,1);
				auto data = data_tmpl;
				data[getIndex(idx[i] == 1 ? 1 : 0,0,0,0)].emplace_back(std::pair<std::vector<size_t>,Tensor>({0},psi0));
				data[getIndex(0,idx[i] == 1 ? 0 : 1,(i+1)%2,i%2)].emplace_back(std::pair<std::vector<size_t>,Tensor>({1},psi1));
				queue.push(std::pair<size_t,std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>>>(i,data));
			}

			bool finished = false;
			size_t count = 0;
			while (not finished){
				finished = queue.size() == 2 ? true : false;
				XERUS_LOG(info, "Queue Size " << queue.size());
				auto elm1 = queue.front();
				queue.pop();
				auto elm2 = queue.front();
				queue.pop();
				if (elm1.first > elm2.first){
					queue.push(elm1);
					elm1 = elm2;
					elm2 = queue.front();
					queue.pop();
				}
				size_t pos = elm1.first;
				XERUS_LOG(info, pos);
				if (pos == 0 ){
					XERUS_LOG(info,elm1.second[getIndex(1,1,2,2)]);
				}

				auto data = data_tmpl;
				for (size_t i1 = 0; i1 < 3; ++i1){
					for (size_t j1 = 0; j1 < 3; ++j1){
						for (size_t k1 = 0; k1 <= p/2; ++k1){
							for (size_t l1 = 0; l1 <= p/2; ++l1){
								for (auto const& tuple1 : elm1.second[getIndex(i1,j1,k1,l1)]){
									for (size_t i2 = 0; i2 < 3-i1; ++i2){
										for (size_t j2 = 0; j2 < 3-j1; ++j2){
											if (not finished){
												for (size_t k2 = 0; k2 <= p/2-k1; ++k2){
													for (size_t l2 = 0; l2 <= p/2-l1; ++l2){
														for (auto const& tuple2 : elm2.second[getIndex(i2,j2,k2,l2)]){
															std::vector<size_t> idx_new(tuple1.first);
															idx_new.insert(idx_new.end(),tuple2.first.begin(),tuple2.first.end());
															Tensor tmp;
//															if (pos == 0 and idx_new.size() <=8){
//																XERUS_LOG(info, i1 << " " <<j1 << " " <<k1 << " " <<l1);
//																XERUS_LOG(info, i2 << " " <<j2 << " " <<k2 << " " <<l2);
//																XERUS_LOG(info, i2+i1 << " " <<j2+j1 << " " <<k2+k1 << " " <<l2+l1);
//																XERUS_LOG(info,idx_new);
//															}
															tmp(r1,r3) = tuple1.second(r1,r2)*tuple2.second(r2,r3);
															data[getIndex(i1+i2,j1+j2,k1+k2,l1+l2)].emplace_back(std::pair<std::vector<size_t>,Tensor>(idx_new,std::move(tmp)));
											}}}}
											else {
												for (auto const& tuple2 : elm2.second[getIndex(i2,j2,p/2-k1,p/2-l1)]){
													std::vector<size_t> idx_new(tuple1.first);
													idx_new.insert(idx_new.end(),tuple2.first.begin(),tuple2.first.end());
													Tensor tmp;
													tmp(r1,r3) = tuple1.second(r1,r2)*tuple2.second(r2,r3);
													umap2_psi[idx] = tmp[0];
													count++;
												}
											}

				}}}}}}}
				if (not finished)
					queue.push(std::pair<size_t,std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>>>(pos,data));

			}
			XERUS_LOG(info, "count " << count);
		}

		// The first index is the number of annihilated particles compared to the sample
		// The second index is the number of created particles compared to the sample
		// The third index is the number of spin up particles contained
		// The fourth index is the number of spin down particles contained
		size_t getIndex(size_t i1, size_t j1, size_t k1, size_t l1){
			return l1 + (p/2+1)*(k1 + (p/2+1)*(j1 + 3*i1));
		}

		value_t returnTValue(size_t p, size_t q)
		{
			if (p > q)
				return T[{p , q }];
			return T[{q , p }];
		}


		value_t returnVValue(size_t i, size_t k, size_t j, size_t l){
			//XERUS_LOG(info, i<<j<<k<<l );
			if (j <= i){
				if (k<= i && l <= (i==k ? j : k))
					return V[{i,j,k ,l}];
				if (l<= i && k <= (i==l ? j : l))
					return V[{i,j,l ,k}];
			} else if (i <= j){
				if (k<= j && l <= (j==k ? i : k))
					return V[{j,i,k,l}];
				if (l<= j && k <= (j==l ? i : l))
					return V[{j,i,l,k}];
			}
			if (l <= k){
				if (i<= k && j <= (k==i ? l : i))
					return V[{k,l,i ,j}];
				if (j<= k && i <= (k==j ? l : j))
					return V[{k,l,j ,i}];
			} else if (k <= l) {
				if (i<= l && j <= (l==i ? k : i))
					return V[{l,k,i ,j}];
				if (j<= l && i <= (l==j ? k : j))
					return V[{l,k,j ,i}];
			}
//			if (j <= i && k<= i && l <= (i==k ? j : k)) //ijkl = 0110
//				return V[{i,j,k ,l}];
//			if (i <= j && k<= j && l <= (j==k ? i : k))
//				return V[{j,i,k,l}];
//			if (j <= i && l<= i && k <= (i==l ? j : l))
//				return V[{i,j,l ,k}];
//			if (i <= j && l<= j && k <= (j==l ? i : l))
//				return V[{j,i,l,k}];
//			if (l <= k && i<= k && j <= (k==i ? l : i))
//				return V[{k,l,i ,j}];
//			if (k <= l && i<= l && j <= (l==i ? k : i))
//				return V[{l,k,i ,j}];
//			if (l <= k && j<= k && i <= (k==j ? l : j))
//				return V[{k,l,j ,i}];
//			if (k <= l && j<= l && i <= (l==j ? k : j))
//				return V[{l,k,j ,i}];
			return 1.0;
		}


		TTTensor getGrad(size_t rank = 0){
			TTTensor res(std::vector<size_t>(d,2));
			value_t signp = 1.0,signq = 1.0,signr =1.0,signs=1.0,val = 0,val1=0,val2 = 0;
			size_t nextp = 0,nextq = 0;
			TTTensor ek;
			size_t count = 0;
			// 1 e contraction
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
				idx[q] = 0; // annil operator a_q
				for (size_t p = 0; p < d; ++p){
					if (idx[p] == 1) {signp *= -1; continue;}
					if (p%2 != q%2) {continue;}
					val = returnTValue(p/2,q/2);
					if (std::abs(val) > 10e-12){
						idx[p] = 1; //creation
						ek = TTTensor::dirac(std::vector<size_t>(d,2),idx);
						count++;
						res += signp *  val * ek;
						idx[p] = 0; //annilation
					}
				}
				signq *= -1;
				signp = signq;
				idx[q] = 1; 				// creation operator a^*q
			}

			signq = 1.0; signr == 1.0; signs = 1.0; signp = 1.0;
			for (size_t r = 0; r < d; ++r){
				if (idx[r] != 1) continue;
				idx[r] = 0;
				signs = signr;
				for (size_t s = 0; s < r; ++s){
					XERUS_LOG(info,"r = " << r << " s = " << s);
					if (rank > 0) res.round(rank);
					if (idx[s] != 1) continue;
					idx[s] = 0;
					signq = signs;
					for (size_t q = 0; q < d; ++q){
						if (idx[q] == 1) {signq *= -1; continue;}
						idx[q] = 1;
						signp = signq;
						for (size_t p = 0; p < q; ++p){
							if (idx[p] == 1) { signp *= -1;  continue;}
							if ((p%2 != r%2 && q%2 != r%2) || (p%2 != s%2 && q%2 != s%2)) {continue;}
							val1 = ((p%2 != r%2) || (q%2!=s%2)) ? 0 : returnVValue(p/2,q/2,r/2,s/2);
							val2 = ((p%2 != s%2) || (q%2!=r%2)) ? 0 : returnVValue(q/2,p/2,r/2,s/2);
							val = signp*(val1 - val2);
							if (std::abs(val) > 10e-12){
								idx[p] = 1;
								ek = TTTensor::dirac(std::vector<size_t>(d,2),idx);
								res += val * ek;
								count++;
								idx[p] = 0;
							}
						}
						idx[q] = 0;
					}
					idx[s] = 1;
					signs *= -1;
				}
				res.round(1e-10);
				idx[r] = 1;
				signr *= -1;
			}
			XERUS_LOG(info,count);
			ek = TTTensor::dirac(std::vector<size_t>(d,2),idx);
			return res  + shift*ek;
		}

		/*
		 * Contraction, does psi ek for current sample, NOTE: use reset first
		 */
		value_t psiEntry(){
			auto itr = umap_psi.find(idx);
			if (itr == umap_psi.end())
				umap_psi[idx] = psi[idx];
			return umap_psi[idx];
		}

		/*
		 * Contraction, returns diagonal entry, NOTE: use reset first
		 */
		value_t diagionalEntry(){
			result = 0;
			value_t val1, val2;
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
					result += returnTValue(q/2,q/2);
			}
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
				for (size_t p = 0; p < d; ++p){
					if (idx[p] != 1 || p == q) continue;
					val1 = returnVValue(p/2,q/2,p/2,q/2);
					val2 = (p%2 != q%2) ? 0 : returnVValue(p/2,q/2,q/2,p/2);
					result += 0.5*(val1-val2);
				}
			}
			return result + shift;
		}
//
//		value_t diagionalEntry2(){
//			result = 0;
//			for (size_t q = 0; q < d; ++q){
//				if (idx[q] != 1) continue;
//					result += T2[{q,q}];
//			}
//			for (size_t q = 0; q < d; ++q){
//				if (idx[q] != 1) continue;
//				for (size_t p = 0; p < d; ++p){
//					if (idx[p] != 1 || p == q) continue;
//						result += 0.5*(V2[{p,q,p,q}]-V2[{p,q,q,p}]);
//				}
//			}
//			return result + shift;
//		}

		value_t evBruteForce(){
			result = 0;
			Tensor tmp;
			value_t val,val1,val2;
			for (size_t q = 0; q < d; ++q){
				for (size_t p = 0; p < d; ++p){
					if (p%2 != q%2) {continue;}
					val = returnTValue(p/2,q/2);
					if (std::abs(val) > 10e-12){
						result += val* contract_TT(return_one_e_ac(p,q,d),psi,psi);
					}
				}
			}

			for (size_t r = 0; r < d; ++r){
				for (size_t s = 0; s < d; ++s){
					XERUS_LOG(info,r << " " << s);
					for (size_t q = 0; q < d; ++q){
						for (size_t p = 0; p < d; ++p){
							if ((p%2 != r%2 && q%2 != r%2) || (p%2 != s%2 && q%2 != s%2)) {continue;}
							val = ((p%2 != r%2) || (q%2!=s%2)) ? 0 : returnVValue(p/2,q/2,r/2,s/2);
							if (std::abs(val) > 10e-12){
								result += 0.5*val* contract_TT(return_two_e_ac(p,q,s,r,d),psi,psi);
							}
						}
					}
				}
			}

			return result +shift;
		}


		void makeInvSampleAndIndex(){
			for (size_t i = 0; i < d; ++i){
				if(!std::binary_search (current_sample.begin(), current_sample.end(), i))
					current_sample_inv.emplace_back(i);
				else
					idx[i] = 1;
			}
		}

		std::vector<size_t> makeIndexToSample(std::vector<size_t> _idx){
			std::vector<size_t> sample;
			for (size_t i = 0; i < _idx.size(); ++i)
				if( idx[i] == 1)
					sample.emplace_back(i);
			return sample;
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

};


