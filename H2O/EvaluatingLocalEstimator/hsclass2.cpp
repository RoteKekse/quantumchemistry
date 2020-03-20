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

class HScontract{
	public:
		const size_t d;
		const size_t p;
		Tensor V;
		Tensor T;
		Tensor N;
		std::string path_T;
		std::string path_V;
		value_t nuc;

	private:
		std::vector<size_t> current_sample;
		std::vector<size_t> current_sample_k;
		std::vector<size_t> current_sample_k_inv;
		TTTensor current_wave_function;
		TTTensor result;
		std::vector<std::vector<TTTensor>> P1;
		std::vector<std::vector<TTTensor>> P2;
		std::vector<std::vector<TTTensor>> Q;
		std::vector<std::vector<bool>> P1nonzero;
		std::vector<std::vector<bool>> P2nonzero;
		std::vector<std::vector<bool>> Qnonzero;
		std::vector<TTTensor> S1;
		std::vector<TTTensor> S2;
		std::vector<bool> S1nonzero;
		std::vector<bool> S2nonzero;
		bool active_site;
	  time_t copy_tt;
	  time_t copy_tt2;
	  time_t one_site;
	  time_t update_res;
	  time_t update_S;
	  time_t update_P;
	  time_t update_Q;
	  time_t update_Q1add;
	  time_t update_Q2add;
	  time_t update_Q1ext;
	  time_t update_Q2ext;
	  time_t build_coeffs;
	  value_t sign;
		TTTensor tmp2;
		TTTensor one_site_template;
		std::vector<TTTensor> one_site_templates;
		int number_of_threads;

	public:
		/*
		 * Constructor
		 */
		HScontract(size_t _d,size_t _p,std::string _path_T, std::string _path_V, value_t _nuc)
		: d(_d), p(_p), path_T(_path_T),path_V(_path_V), nuc(_nuc){
			XERUS_LOG(HScontract, "---- Initializing Building Operator L2R ----");
			XERUS_LOG(HScontract, "- Loading 1e and 2e Integrals and Nuclear Potential");
			T = load1eIntegrals();
			V = load2eIntegrals();
			N = loadNuclear();
			copy_tt = 0;
			copy_tt2 = 0;
			one_site = 0;
			update_res = 0;
			update_S = 0;
			update_P = 0;
			update_Q = 0;
			update_Q1add = 0;
			update_Q1ext = 0;
			build_coeffs = 0;
			number_of_threads = 4;
		}

		/*
		 * Build Operator
		 */
		TTTensor contract(std::vector<size_t> sample){
			clearVectors();
			current_sample = sample;
			current_wave_function = makeUnitVector(current_sample);
			active_site = std::binary_search (current_sample.begin(), current_sample.end(), 0);
			if (active_site) current_sample_k.emplace_back(0);
			else current_sample_k_inv.emplace_back(0);
			iniOneSiteTemplate();
			iniResult();
			ini3SiteOperatorsS();
			ini2SiteOperatorQ();
			ini2SiteOperatorPs();
			sign = 1.0;
			for (size_t k = 1; k < d; ++k){
				if (active_site) sign *= -1; // if the last was active change sign!
				active_site = std::binary_search (current_sample.begin(), current_sample.end(), k);
				step(k);
				if (active_site) current_sample_k.emplace_back(k);
				else current_sample_k_inv.emplace_back(k);
			}
			XERUS_LOG(debug, "Time to update result:   " << update_res << " sec");
			XERUS_LOG(debug, "Time to update S:        " << update_S << " sec");
			XERUS_LOG(debug, "Time to update P:        " << update_P << " sec");
			XERUS_LOG(debug, "Time to update Q:        " << update_Q << " sec");
			XERUS_LOG(debug, "Time to update Q1add:    " << update_Q1add << " sec");
			XERUS_LOG(debug, "Time to update Q1ext:    " << update_Q1ext << " sec");
			XERUS_LOG(debug, "Time to add identity1:   " << copy_tt << " sec");
			XERUS_LOG(debug, "Time to add identity2:   " << copy_tt2 << " sec");
			XERUS_LOG(debug, "Time to contract 1 site: " << one_site << " sec");
			XERUS_LOG(debug, "Time to build coeffs:    " << build_coeffs << " sec");
			XERUS_LOG(debug, "sample: " << current_sample);
			return result;
		}

		/*
		 * One step from k to k + 1
		 */
		void step(const size_t k){
			XERUS_LOG(debug, "- Step " << k);
			time_t begin_time = time (NULL);
			updateOneSiteTemplate(k+1);
#if test
			testResult(k);
#endif
			updateResult(k+1);
			update_res += time (NULL) - begin_time;
			begin_time = time (NULL);
#pragma omp parallel for num_threads(number_of_threads)
			for (size_t s = k; s < d; ++s){
#if test
				testS1(s,k);
				testS2(s,k);
#endif
				update3SiteOperatorsS(s,k+1);
#if test2
				auto tt = makeTT(S1[s],k+1);
				auto norm = tt.frob_norm();
				if (norm < 10e-12 and tt.ranks()[0] > 1)
					XERUS_LOG(debug,"S1 " <<s <<" " << norm << " " << tt.ranks());
				tt = makeTT(S1[s],k+1);
				norm = tt.frob_norm();
				if (norm < 10e-12 and tt.ranks()[0] > 1)
					XERUS_LOG(debug,"S2 " <<s <<" " << norm << " " << tt.ranks());
#endif
			}
			update_S += time (NULL) - begin_time;
			begin_time = time (NULL);
#pragma omp parallel for  num_threads(number_of_threads)
			for (size_t r = k; r < d; ++r){
				for (size_t s = k+1; s < d; ++s){
#if test
					testP1(r,s,k);
					testP2(r,s,k);
#endif
					update2SiteOperatorsP(r,s,k+1);
//			    auto r1 = P1[r][s].ranks();
//					auto r2 = P2[r][s].ranks();
//			    auto max1 = std::max_element(r1.begin(), r1.end());
//			    auto max2 = std::max_element(r2.begin(), r2.end());
					//if (max1[0] >= 6) XERUS_LOG(info, "P1 dim = " << k+1 << " l = " << r << " m = " << s << P1[r][s].ranks());
					//if (max2[0] >= 20) XERUS_LOG(info, "P2 dim = " << k+1 << " l = " << r << " m = " << s << P2[r][s].ranks());
#if test2
					auto tt = makeTT(P1[r][s],k+1);
					auto norm = tt.frob_norm();
					if (norm < 10e-12 and tt.ranks()[0] > 1)
						XERUS_LOG(debug,"P1 " << r << " "<<s <<" " << norm << " " << tt.ranks());
					tt = makeTT(P2[r][s],k+1);
					norm = tt.frob_norm();
					if (norm < 10e-12 and tt.ranks()[0] > 1)
						XERUS_LOG(debug,"P2 " <<r << " "<<s <<" " << norm << " " << tt.ranks());
#endif
				}
			}
			update_P += time (NULL) - begin_time;
			begin_time = time (NULL);
#pragma omp parallel for num_threads(number_of_threads)
			for (size_t r = k; r < d; ++r){
				for (size_t s = k; s < d; ++s){
#if test
					testQ(r,s,k);
#endif
					update2SiteOperatorQ(r,s,k+1);
#if test2
					auto tt = makeTT(Q[r][s],k+1);
					auto norm = tt.frob_norm();
					if (norm < 10e-12 and tt.ranks()[0] > 1)
						XERUS_LOG(debug,"Q " <<r << " "<<s <<" " << norm << " " << tt.ranks());
#endif
				}
			}
			update_Q += time (NULL) - begin_time;
		}


		void iniOneSiteTemplate(){
			one_site_template = TTTensor(std::vector<size_t>(d,2));
			for (size_t i = 0; i < number_of_threads; ++i)
				one_site_templates.emplace_back(one_site_template);
		}
		/*
		 * initializes the Result (k = 0)
		 */
		void iniResult(){
			result = TTTensor(std::vector<size_t>(d,2));
			if (active_site)
				result.component(0)[{0,1,0}] = T[{0,0}];
		}

		/*
		 * initializes operator S1 and S2 (k = 0)
		 */
		void ini3SiteOperatorsS(){
			for(size_t l = 0; l < d; ++l){ //k<l
				S1.emplace_back(TTTensor(std::vector<size_t>(d,2)));
				S2.emplace_back(TTTensor(std::vector<size_t>(d,2)));
				S1nonzero.emplace_back(false);
				S2nonzero.emplace_back(false);

				if (l == 0) continue;
				value_t val = T[{l,0}];
				if (active_site and std::abs(val) > 10e-12){
					S1[l].component(0)[{0,0,0}] = val;
					S1nonzero[l] = true;
				}
				else if (!active_site and std::abs(val) > 10e-12){
					S2[l].component(0)[{0,1,0}] = val;
					S2nonzero[l] = true;
				}
			}
		}

		/*
		 * initializes operator Q (k = 0)
		 */
		void ini2SiteOperatorQ(){
			for (size_t l = 0; l < d; ++l){
				std::vector<TTTensor> tmp2;
				std::vector<bool> tmp3;
				Q.emplace_back(tmp2);
				Qnonzero.emplace_back(tmp3);
				if (l == 0) continue;
				for(size_t m = 0; m < d; ++m){
					Q[l].emplace_back(TTTensor(std::vector<size_t>(d,2)));
					Qnonzero[l].emplace_back(false);
					if (m == 0) continue;
					value_t val = V[{l,0,m,0}] - V[{0,l,m,0}];
					if (active_site and std::abs(val) > 10e-12){
						Q[l][m].component(0)[{0,1,0}] = val;
						Qnonzero[l][m] = true;
					}
				}
			}
		}

		/*
		 * initializes operator P1,P2 (k = 0)
		 */
		void ini2SiteOperatorPs(){
			for (size_t l = 0; l < d; ++l){
				std::vector<TTTensor> tmp2,tmp3;
				std::vector<bool> tmp2b,tmp3b;
				P1.emplace_back(tmp2);
				P1nonzero.emplace_back(tmp2b);
				P2.emplace_back(tmp3);
				P2nonzero.emplace_back(tmp3b);
				for(size_t m = 0; m < d; ++m){
					P1[l].emplace_back(TTTensor(std::vector<size_t>(d,2)));
					P1nonzero[l].emplace_back(false);
					P2[l].emplace_back(TTTensor(std::vector<size_t>(d,2)));
					P2nonzero[l].emplace_back(false);
				}
			}
		}

		void clearVectors(){
			one_site_templates.clear();
			current_sample.clear();
			current_sample_k.clear();
			current_sample_k_inv.clear();
			P1.clear();
			P1nonzero.clear();
			P2.clear();
			P2nonzero.clear();
			S1.clear();
			S1nonzero.clear();
			S2.clear();
			S2nonzero.clear();
			Q.clear();
			Qnonzero.clear();
		}

		void updateOneSiteTemplate(size_t dim){
			Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});
			for (size_t site = 0; site < dim - 1; ++ site){
				bool active_s = std::binary_search (current_sample.begin(), current_sample.end(), site);
				if (site == 0){
					Tensor tmp;
					if (dim == 2)
						tmp = Tensor({1,2,1});
					else{
						tmp = Tensor({1,2,2});
						tmp.offset_add(active_s ? (-1)*e1 : e0,{0,0,0});
					}
					one_site_template.set_component(site,tmp);
				} else if (site == dim - 2) {
					Tensor tmp({2,2,1});
					tmp.offset_add(active_s ? e1 : e0,{1,0,0});
					one_site_template.set_component(site,tmp);
				} else {
					Tensor tmp({2,2,2});
					tmp.offset_add(active_s ? (-1)*e1 : e0,{0,0,0});
					tmp.offset_add(active_s ? e1 : e0,{1,0,1});
					one_site_template.set_component(site,tmp);
				}

			}
			for (size_t i = 0; i < number_of_threads; ++i)
				one_site_templates[i] = one_site_template;
		}

		/*
		 * Builds the hamiltonian from left to right, uses S and Q
		 * overwrites H from the last site
		 */
		void updateResult(const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;
			const Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			const Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});
			//first summand
			result.set_component(k, current_wave_function.get_component(k));
			if (active_site){

				if (std::abs(T[{k,k}]) > 10e-12) add_TT(result,addNextSiteIter(current_wave_function,T[{k,k}]*e1,k),dim);
				if (S2nonzero[k]){
					S2[k].set_component(k,e0); //TODO check why is S2 0?
					add_TT(result,sign*S2[k],dim);
				}
				if (Qnonzero[k][k]){
					Q[k][k].set_component(k,e1);
					add_TT(result,Q[k][k],dim);
				}


			} else {
				if (S1nonzero[k]){
					S1[k].set_component(k,e1);
					add_TT(result,(-1)*sign*S1[k],dim);
				}
			}
			if (dim%2 == 0 || dim == d) partial_round(result,dim);
			XERUS_LOG(debug, result.ranks());

		}

		/*
		 * Builds the sum over three indices, sum_pqr w_prqs a_p^* a_q^* a_r and includes 1e parts
		 * overwrites S from the last site
		 */
		void update3SiteOperatorsS(const size_t l, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;
			std::queue<value_t> coeffs1;
			std::queue<value_t> coeffs2;
			const Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			const Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});
			S1[l].set_component(k,current_wave_function.get_component(k));
			S2[l].set_component(k,current_wave_function.get_component(k));
			if (active_site){
//			tests1(coeffs1,k,false);
				bool nonzero = pushVCoeffs(coeffs1,k,k,l,k,k,k,l,false);
				bool nonzero1 = pushVCoeffs(coeffs2,k,k,l,k,k,k,l,true);
//				tests1(coeffs2,k,true);

				if (std::abs(T[{l,k}]) > 10e-12) {
					add_TT(S1[l],sign*addNextSiteIter(current_wave_function,T[{l,k}]*e0,k),dim);
					if (!S1nonzero[l])
						S1nonzero[l] = true;
				}
				if (Qnonzero[l][k]){
					Q[l][k].set_component(k,e0);
					add_TT(S1[l],sign*Q[l][k],dim);
					if (!S1nonzero[l])
						S1nonzero[l] = true;
				}
				if (nonzero){
					add_TT(S1[l],build1SiteTensor(coeffs1, e1, dim, false),dim);
					if (!S1nonzero[l])
						S1nonzero[l] = true;
				}

				if (P2nonzero[l][k]){
					P2[l][k].set_component(k,e0);
					add_TT(S2[l],sign* P2[l][k],dim);
					if (!S2nonzero[l])
						S2nonzero[l] = true;
				}
				if (nonzero1){
					add_TT(S2[l],build1SiteTensor(coeffs2, e1, dim, true),dim);
					if (!S2nonzero[l])
						S2nonzero[l] = true;
				}

			} else {
				if (P1nonzero[l][k]){
					P1[l][k].set_component(k,e1);
					add_TT(S1[l],sign*P1[l][k],dim);
					if (!S1nonzero[l])
						S1nonzero[l] = true;
				}
				if (std::abs(T[{l,k}]) > 10e-12) {
					add_TT(S2[l],sign*addNextSiteIter(current_wave_function,T[{l,k}]*e1,k),dim);
					if (!S2nonzero[l])
						S2nonzero[l] = true;
				}

				if (Qnonzero[k][l]){
					Q[k][l].set_component(k,e1);
					add_TT(S2[l],sign*Q[k][l],dim);
					if (!S2nonzero[l])
						S2nonzero[l] = true;
				}

			}
			if (k%3 == 0){
				partial_round(S1[l],dim);
				partial_round(S2[l],dim);
				//S1[l].round(0.0);
				//S2[l].round(0.0);
			}
		}

		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q^* a_p
		 * overwrites Q from the last site
		 */
		void update2SiteOperatorQ(const size_t l, const size_t m, const size_t dim){// TODO check if Q1 and Q2 are definitively different
			Index ii,jj,kk;
			size_t k = dim - 1;
			std::queue<value_t> coeffs1;
			const Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			const Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});
			//first summand, til k - 1
			Q[l][m].set_component(k,current_wave_function.get_component(k));
			bool change = false;
			if (active_site){
				bool nonzero = pushVCoeffs(coeffs1,k,l,k,m,k,m,l,true);

				value_t val = V[{l,k,m,k}]-V[{k,l,m,k}];
				if (std::abs(val) > 10e-12){
					add_TT(Q[l][m],addNextSiteIter(current_wave_function,val*e1,k),dim);
					change = true;
					if (!Qnonzero[l][m])
						Qnonzero[l][m] = true;
				}
				if (nonzero){
					add_TT(Q[l][m],sign*build1SiteTensor(coeffs1, e0, dim, true),dim);
					change = true;
					if (!Qnonzero[l][m])
						Qnonzero[l][m] = true;
				}

			} else {

				bool nonzero = pushVCoeffs(coeffs1,k,l,k,m,k,l,m,false);
				if (nonzero){
					add_TT(Q[l][m],(-1)*sign*build1SiteTensor(coeffs1, e1,dim, false),dim);
					change = true;
					if (!Qnonzero[l][m])
						Qnonzero[l][m] = true;
				}

			}
			if (change && k%5 == 0){
				partial_round(Q[l][m],dim);
			}
		}


		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q a_p
		 * overwrites P from the last site
		 */
		void update2SiteOperatorsP(const size_t l, const size_t m, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;
			std::queue<value_t> coeffs1;
			const Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			const Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});

			bool change = false;

			//first summand, til k - 1, Note there are no summands for only since they are 0
			P1[l][m].set_component(k,current_wave_function.get_component(k));
			P2[l][m].set_component(k,current_wave_function.get_component(k));

			if (active_site){
				bool nonzero = pushVCoeffs(coeffs1,k,k,l,m,k,m,l,false);
				if (nonzero) {
					add_TT(P1[l][m],(-1)*sign*build1SiteTensor(coeffs1, e0, dim, false),dim);
					change = true;
					if (!P1nonzero[l][m])
						P1nonzero[l][m] = true;
				}

			} else {
				bool nonzero = pushVCoeffs(coeffs1,k,k,m,l,k,l,m,true);
				if (nonzero){
					add_TT(P2[l][m],(-1)*sign*build1SiteTensor(coeffs1, e1, dim, true),dim);
					change = true;
					if (!P2nonzero[l][m])
						P2nonzero[l][m] = true;
				}
			}
			if (change && k%5 == 0){
				if (active_site)
					partial_round(P1[l][m],dim);
				else
					partial_round(P2[l][m],dim);
			}
		}

		bool pushVCoeffs(std::queue<value_t> &coeffs,size_t k, size_t i1,size_t i2,size_t i3, size_t j1, size_t j2, size_t j3, bool inv){
			time_t begin_time = time (NULL);
			bool nonzero = false;
			value_t val;
			for (size_t s : (inv ? current_sample_k_inv : current_sample_k)){
				val = V[{s,i1,i2,i3}] - V[{s,j1,j2,j3}];
				if (std::abs(val) > 10e-12)
					nonzero = true;
				coeffs.push(val);
			}
			build_coeffs += time(NULL) - begin_time;
//		XERUS_LOG(debug, nonzero);
			return nonzero;
		}

		/*
		 * Builds the sum over one index, sum_p w_prqs a_p, as a rank 2 operator
		 */
		TTTensor build1SiteTensor(std::queue<value_t> coeffs, Tensor next_comp, const size_t dim, const bool transpose){

			XERUS_REQUIRE(coeffs.size() <= dim,"Number of coefficients is larger than dimension of result");

			time_t begin_time = time (NULL);
			int thread_id = omp_get_thread_num();

			auto coeffs2 = coeffs;
			//while (!coeffs.empty()){
			for (size_t site = 0; site < dim -1 ; ++site){
				value_t coeff = 0;
				bool active_s = std::binary_search (current_sample.begin(), current_sample.end(), site);
				if (!transpose == active_s){
					coeff = coeffs.front();
					coeffs.pop();
				}
				if (site == 0){
					if (dim <= 2){
						if (!transpose == active_s){
							one_site_templates[thread_id].component(site)[{0,0,0}] = transpose ? 0 : coeff;
							one_site_templates[thread_id].component(site)[{0,1,0}] = transpose ? coeff : 0;
						} else {
							one_site_templates[thread_id].component(site)[{0,0,0}] = 0.0;
							one_site_templates[thread_id].component(site)[{0,1,0}] = 0.0;
						}
					} else {
						if (!transpose == active_s){
							one_site_templates[thread_id].component(site)[{0,0,1}] = transpose ? 0 : coeff;
							one_site_templates[thread_id].component(site)[{0,1,1}] = transpose ? coeff : 0;
						} else{
							one_site_templates[thread_id].component(site)[{0,0,1}] = 0.0;
							one_site_templates[thread_id].component(site)[{0,1,1}] = 0.0;
						}
					}
				} else if (site == dim - 2) {
						if (!transpose == active_s){
							one_site_templates[thread_id].component(site)[{0,0,0}] = transpose ? 0 : coeff;
							one_site_templates[thread_id].component(site)[{0,1,0}] = transpose ? coeff : 0;
						} else{
							one_site_templates[thread_id].component(site)[{0,0,0}] = 0.0;
							one_site_templates[thread_id].component(site)[{0,1,0}] = 0.0;
						}
				} else {
					if (!transpose == active_s){
						one_site_templates[thread_id].component(site)[{0,0,1}] = transpose ? 0 : coeff;
						one_site_templates[thread_id].component(site)[{0,1,1}] = transpose ? coeff : 0;
					} else{
						one_site_templates[thread_id].component(site)[{0,0,1}] = 0.0;
						one_site_templates[thread_id].component(site)[{0,1,1}] = 0.0;
					}
				}
				//++site;
			}

			one_site_templates[thread_id].set_component(dim - 1, next_comp);
			//result.round(0.0); //TODO necessary ?
			one_site += time(NULL) - begin_time;

			return one_site_templates[thread_id];
		}

		/*
		 * TODO These can be improved with better storage management!
		 */
		// Takes the given operator and adds the next site to the left
		TTTensor addNextSiteIter(TTTensor tensor_train, Tensor next_comp,size_t k){
			time_t begin_time = time (NULL);
			TTTensor result_tmp(k+1);
			for (size_t i =  0; i < k; ++i){
				result_tmp.set_component(i,tensor_train.get_component(i)); //TODO check if component works instead of get_component
			}
			result_tmp.set_component(k, next_comp);
			copy_tt += time(NULL) - begin_time;
			return result_tmp;
		}

		void partial_round(TTTensor & tt, size_t dim){
			for (size_t n = 0; n < dim-1; ++n) {
				tt.transfer_core(n+1, n+2, true);
			}
			auto epsPerSite = 10e-14;
			for(size_t i = 0; i+1 < dim; ++i) {
				tt.round_edge(dim-i, dim-i-1, std::vector<size_t>(tt.ranks().size(), std::numeric_limits<size_t>::max())[dim-i-2], epsPerSite, 0.0);
			}
		}

		void add_TT(TTTensor &tt1, const TTTensor &tt2, size_t dim){
			XERUS_PA_START;
			for(size_t position = 0; position < dim; ++position) {
				// Get current components
				const Tensor& myComponent = tt1.get_component(position);
				const Tensor& otherComponent = tt2.get_component(position);

				// Structure has to be (for order 4)
				// (L1 R1) * ( L2 0  ) * ( L3 0  ) * ( L4 )
				//           ( 0  R2 )   ( 0  R3 )   ( R4 )

				// Create a Tensor for the result
				std::vector<size_t> nxtDimensions;
				nxtDimensions.emplace_back(position == 0 ? 1 : myComponent.dimensions.front()+otherComponent.dimensions.front());
				nxtDimensions.emplace_back(myComponent.dimensions[1]);
				nxtDimensions.emplace_back(position == dim-1 ? 1 : myComponent.dimensions.back()+otherComponent.dimensions.back());

				const Tensor::Representation newRep = myComponent.is_sparse() && otherComponent.is_sparse() ? Tensor::Representation::Sparse : Tensor::Representation::Dense;
				std::unique_ptr<Tensor> newComponent(new Tensor(std::move(nxtDimensions), newRep));

				newComponent->offset_add(myComponent, std::vector<size_t>({0,0,0}));

				const size_t leftOffset = position == 0 ? 0 : myComponent.dimensions.front();
				const size_t rightOffset = position == dim-1 ? 0 : myComponent.dimensions.back();

				newComponent->offset_add(otherComponent, std::vector<size_t>({leftOffset,0,rightOffset}));

				tt1.set_component(position, std::move(*newComponent));
			}
			XERUS_PA_END("ADD/SUB", "TTNetwork ADD/SUB", std::string("Dims:")+misc::to_string(dimensions)+" Ranks: "+misc::to_string(ranks()));
		}


		TTTensor makeTT(TTTensor tt1,size_t dim){
			TTTensor result(std::vector<size_t>(dim,2));
			for (size_t i =  0; i < dim; ++i){
				result.set_component(i,tt1.get_component(i)); //TODO check if component works instead of get_component
			}
			return result;
		}


		void testResult(const size_t dim){
			Index ii,jj;

			auto brute = build_brute(dim);

			auto unit_tmp = makeUnitVector(current_sample, dim);
			TTTensor result_tmp;
			result_tmp(ii&0) = brute(ii/2,jj/2) * unit_tmp(jj&0);
			auto norm = (makeTT(result,dim) - result_tmp).frob_norm();
			XERUS_LOG(debug, "Test Result dim = " << dim << " error = " << (norm < 10e-14 ? 0 : norm));

		}


		void testS1(const size_t l, const size_t dim){
			Index ii,jj;
			TTTensor tmpS1_b(std::vector<size_t>(dim,2));
			TTTensor tmp11,tt_tmp;

			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t rr = 0; rr < dim;++rr){
				auto tmp1 = T[{l,rr}]*return_annil(rr, dim);
				tmp11(ii&0) = tmp1(ii/2,jj/2) * unit_tmp(jj&0);
				tmpS1_b += tmp11;
				for(size_t ss = 0; ss < dim; ++ss){
					for(size_t pp = 0; pp < dim; ++pp){
						auto tmp = 0.5*(V[{l,pp,rr,ss}] - V[{pp,l,rr,ss}])*return_two_e_ac_partial2(pp, ss,rr, dim);
						tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
						tmpS1_b += tt_tmp;
					}
				}
			}
			auto norm = (makeTT(S1[l],dim) - tmpS1_b).frob_norm();
			if (norm > 10e-12)
				XERUS_LOG(debug, "Test S1 dim = " << dim << " and l = " << l << "           error = " << (norm < 10e-14 ? 0 : norm));
		}



		void testS2(const size_t l, const size_t dim){
			Index ii,jj;
			TTTensor tmpS2_b(std::vector<size_t>(dim,2));
			TTTensor tmp11,tt_tmp;
			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t rr = 0; rr < dim;++rr){
				auto tmp1 = T[{rr,l}]*return_create(rr, dim);
				tmp11(ii&0) = tmp1(ii/2,jj/2) * unit_tmp(jj&0);
				tmpS2_b += tmp11;
				for(size_t pp = 0; pp < dim; ++pp){
					for(size_t qq = 0; qq < dim; ++qq){
						auto tmp = 0.5*(V[{pp,qq,l,rr}] - V[{pp,qq,rr,l}])*return_two_e_ac_partial(pp, qq,rr, dim);
						tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
						tmpS2_b += tt_tmp;
					}
				}
			}
			auto norm = (makeTT(S2[l],dim) - tmpS2_b).frob_norm();
			if (norm > 10e-12)
				XERUS_LOG(debug, "Test S2 dim = " << dim << " and l = " << l << "           error = " << (norm < 10e-14 ? 0 : norm));
		}
		void testQ(const size_t l, size_t m, const size_t dim){
			Index ii,jj;
			TTTensor tmpQ1_b(std::vector<size_t>(dim,2));
			TTTensor tt_tmp;

			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t ss = 0; ss < dim;++ss){
				for(size_t pp = 0; pp < dim; ++pp){
					auto tmp = (V[{l,pp,m,ss}] - V[{pp,l,m,ss}])*return_one_e_ac(pp, ss, dim);
					tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
					tmpQ1_b += tt_tmp;
				}
			}
			auto norm = (makeTT(Q[l][m],dim) - tmpQ1_b).frob_norm();
			if (norm > 10e-12)
				XERUS_LOG(debug, "Test Q  dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << (norm < 10e-14 ? 0 :norm));
		}


		void testP1(const size_t l, size_t m, const size_t dim){
			Index ii,jj;
			TTTensor tmpP1_b(std::vector<size_t>(dim,2));
			TTTensor tt_tmp;
			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t ss =0; ss < dim;++ss){
				for(size_t rr = 0; rr < dim; ++rr){
					auto tmp = 0.5*(V[{l,m,rr,ss}] - V[{m,l,rr,ss}])*return_two_e_ac_partial4(ss, rr, dim);
					TTTensor tmp2;
					tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
					tmpP1_b += tt_tmp;
				}
			}
			auto norm = (makeTT(P1[l][m],dim) - tmpP1_b).frob_norm();
			if (norm > 10e-12)
				XERUS_LOG(debug, "Test P2 dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << (norm< 10e-14 ? 0 :norm));		}

		void testP2(const size_t l, size_t m, const size_t dim){
			Index ii,jj;
			TTTensor tmpP2_b(std::vector<size_t>(dim,2));
			TTTensor tt_tmp;
			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t pp =0; pp < dim;++pp){
				for(size_t qq = 0; qq < dim; ++qq){
					auto tmp = 0.5*(V[{pp,qq,l,m}] - V[{pp,qq,m,l}])*return_two_e_ac_partial3(pp, qq, dim);
					tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
					tmpP2_b += tt_tmp;
				}
			}
			auto norm = (makeTT(P2[l][m],dim) - tmpP2_b).frob_norm();
			if (norm > 10e-12)
				XERUS_LOG(debug, "Test P2 dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << (norm< 10e-14 ? 0 :norm));
		}




		// Takes the given operator and adds an identity to the right or left
		TTOperator add_identity(TTOperator A, bool left=false){
			TTOperator tmpA(A.order() + 2);
			for (size_t i = (left ? 1 : 0); i < A.order() / 2 + (left ? 1 : 0); ++i){
				tmpA.set_component(i,A.get_component(left ? i - 1 : i)); //TODO check if component works instead of get_component
			}
			Tensor id = Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			tmpA.set_component(left ? 0 : A.order() / 2,id);
			return tmpA;
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
					for (size_t j = 0; j < dim; j++){
							value_t val = T[{i , j}];
							auto one_e_ac = return_one_e_ac(i,j,dim);
							opH += val * one_e_ac;
					}
			}

			auto opV = xerus::TTOperator(std::vector<size_t>(2*dim,2));
			for (size_t i = 0; i < dim; i++){
				for (size_t j = 0; j < dim; j++){
					for (size_t k = 0; k < dim; k++){
						for (size_t l = 0; l < dim; l++){
							value_t val = V[{i,j,k,l}];

							auto two_e_ac = return_two_e_ac(i,j,l,k,dim);
							if (std::abs(val) < 10e-14 || i == j || k == l )//|| a == b || c: == d) // TODO check i == j || k == l
									continue;
							opV += 0.5*val * two_e_ac; // TODO: Note change back swap k,l
						}
					}
				}
			}
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


