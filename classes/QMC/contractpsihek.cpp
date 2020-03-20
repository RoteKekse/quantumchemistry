#include <xerus.h>
#include <experimental/random>
//#include "ttcontainer.h"
#include <queue>
#include <omp.h>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <unordered_map>
#include "../classes/containerhash.cpp"

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
		const size_t particle;
		Tensor V;
		Tensor T;
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
		std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> umap_psi; // storing evaluations of psi

	public:
		/*
		 * Constructor
		 */
		ContractPsiHek(TTTensor _psi,size_t _d,size_t _p,std::string _path_T, std::string _path_V, value_t _nuc, value_t _shift)
		: psi(_psi), d(_d), particle(_p), path_T(_path_T),path_V(_path_V), nuc(_nuc), idx(d,0), shift(_shift){
			T = load1eIntegrals();
			V = load2eIntegrals();
			N = loadNuclear();
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

			signq = 1.0; signr == 1.0; signs = 1.0; signp = 1.0;
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
							if ((p%2 != r%2 && q%2 != r%2) || (p%2 != s%2 && q%2 != s%2)) {continue;}
							val = signp*(V[{p,q,r,s}] - V[{q,p,r,s}]);
							if (std::abs(val) > 10e-12){
								idx[p] = 1;
								auto itr = umap_psi.find(idx);
								if (itr == umap_psi.end())
									umap_psi[idx] = psi[idx];
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
			return result + shift * psiEntry();
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
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
					result += T[{q,q}];
			}
			for (size_t q = 0; q < d; ++q){
				if (idx[q] != 1) continue;
				for (size_t p = 0; p < d; ++p){
					if (idx[p] != 1 || p == q) continue;
						result += 0.5*(V[{p,q,p,q}]-V[{p,q,q,p}]);
				}
			}
			return result + shift;
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


