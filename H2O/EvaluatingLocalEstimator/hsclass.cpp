#include <xerus.h>
#include <experimental/random>

#include <queue>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

#include <memory>

#define debug 0
#define test 0

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
		TTTensor current_wave_function;
		TTTensor result;
		std::vector<std::vector<TTTensor>> P1;
		std::vector<std::vector<TTTensor>> P2;
		std::vector<std::vector<TTTensor>> Q1;
		std::vector<std::vector<TTTensor>> Q2;
		std::vector<TTTensor> S1;
		std::vector<TTTensor> S2;
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
	  value_t sign;
		TTTensor tmp2;


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
			update_Q2add = 0;
			update_Q1ext = 0;
			update_Q2ext = 0;
		}

		/*
		 * Build Operator
		 */
		TTTensor contract(std::vector<size_t> sample){
			XERUS_LOG(HScontract, "---- Building Operator ----");
			current_sample = sample;
			current_wave_function = makeUnitVector(current_sample);
			XERUS_LOG(HScontract,"initialize S1,S2,Q,P1,P2 and result");
			active_site = std::binary_search (current_sample.begin(), current_sample.end(), 0);
			iniResult();
			ini3SiteOperatorsS();
			ini2SiteOperatorQ();
			ini2SiteOperatorPs();
			sign = 1.0;
			for (size_t k = 1; k < d; ++k){
				if (active_site) sign *= -1; // if the last was active change sign!
				active_site = std::binary_search (current_sample.begin(), current_sample.end(), k);
				step(k);
				XERUS_LOG(HScontract, "- Ranks of result " << result.ranks());
			}

			XERUS_LOG(info, "Time to update result:   " << update_res << " sec");
			XERUS_LOG(info, "Time to update S:        " << update_S << " sec");
			XERUS_LOG(info, "Time to update P:        " << update_P << " sec");
			XERUS_LOG(info, "Time to update Q:        " << update_Q << " sec");
			XERUS_LOG(info, "Time to update Q1add:    " << update_Q1add << " sec");
			XERUS_LOG(info, "Time to update Q2add:    " << update_Q2add << " sec");
			XERUS_LOG(info, "Time to update Q1ext:    " << update_Q1ext << " sec");
			XERUS_LOG(info, "Time to update Q2ext:    " << update_Q2ext << " sec");
			XERUS_LOG(info, "Time to add identity1:   " << copy_tt << " sec");
			XERUS_LOG(info, "Time to add identity2:   " << copy_tt2 << " sec");
			XERUS_LOG(info, "Time to contract 1 site: " << one_site << " sec");

			return result;
		}

		/*
		 * One step from k to k + 1
		 */
		void step(const size_t k){
			XERUS_LOG(HScontract, "- Step " << k);
			time_t begin_time = time (NULL);
#if test
			testResult(k);
#endif
			updateResult(k+1);
			update_res += time (NULL) - begin_time;
			begin_time = time (NULL);
//#pragma omp parallel for
			for (size_t s = k; s < d; ++s){
#if test
				testS1(s,k);
				testS2(s,k);
#endif
				update3SiteOperatorsS(s,k+1);
			}
			update_S += time (NULL) - begin_time;
			begin_time = time (NULL);
//#pragma omp parallel for
			for (size_t r = k; r < d; ++r){
				for (size_t s = k; s < d; ++s){
#if test
					testP1(r,s,k);
					testP2(r,s,k);
#endif
					update2SiteOperatorsP(r,s,k+1);
				}
			}
			update_P += time (NULL) - begin_time;
			begin_time = time (NULL);
//#pragma omp parallel for
			for (size_t r = k; r < d; ++r){
				for (size_t s = k; s < d; ++s){
#if test
					testQ1(r,s,k);
					testQ2(r,s,k);
#endif
					update2SiteOperatorQ(r,s,k+1);
				}
			}
			update_Q += time (NULL) - begin_time;




		}

		/*
		 * initializes the Result (k = 0)
		 */
		void iniResult(){
			result = TTTensor(std::vector<size_t>(1,2));
			if (active_site)
				result.component(0)[{0,1,0}] = T[{0,0}];
		}

		/*
		 * initializes operator S1 and S2 (k = 0)
		 */
		void ini3SiteOperatorsS(){
			for(size_t l = 0; l < d; ++l){ //k<l
				S1.emplace_back(TTTensor(std::vector<size_t>(1,2)));
				S2.emplace_back(TTTensor(std::vector<size_t>(1,2)));
				if (l == 0) continue;
				if (active_site)
					S1[l].component(0)[{0,0,0}] = T[{l,0}];
				else
					S2[l].component(0)[{0,1,0}] = T[{0,l}];
			}
		}

		/*
		 * initializes operator Q (k = 0)
		 */
		void ini2SiteOperatorQ(){
			for (size_t l = 0; l < d; ++l){
				std::vector<TTTensor> tmp2;
				std::vector<TTTensor> tmp3;
				Q1.emplace_back(tmp2);
				Q2.emplace_back(tmp2);
				if (l == 0) continue;
				for(size_t m = 0; m < d; ++m){
					Q1[l].emplace_back(TTTensor(std::vector<size_t>(1,2)));
					Q2[l].emplace_back(TTTensor(std::vector<size_t>(1,2)));
					if (m == 0) continue;
					if (active_site){
						Q1[l][m].component(0)[{0,1,0}] = V[{l,0,m,0}] - V[{0,l,m,0}];
						Q2[l][m].component(0)[{0,1,0}] = V[{m,0,l,0}] - V[{m,0,0,l}];
					}
				}
			}
		}

		/*
		 * initializes operator P1,P2 (k = 0)
		 */
		void ini2SiteOperatorPs(){
			for (size_t l = 0; l < d; ++l){
				std::vector<TTTensor> tmp2;
				std::vector<TTTensor> tmp3;
				P1.emplace_back(tmp2);
				P2.emplace_back(tmp3);
				for(size_t m = 0; m < d; ++m){
					P1[l].emplace_back(TTTensor(std::vector<size_t>(1,2)));
					P2[l].emplace_back(TTTensor(std::vector<size_t>(1,2)));

				}
			}
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
			addNextSite(result, current_wave_function.get_component(k),k);
			if (active_site){
				addNextSite(Q1[k][k],e1,k);
				addNextSite(S2[k],e0,k); //TODO check why is S2 0?
				auto tmpk = addNextSiteIter(current_wave_function,T[{k,k}]*e1,k);

				result += tmpk + sign*S2[k] +  Q1[k][k];


			} else {
				addNextSite(S1[k],e1,k);
				result -= sign*S1[k];
			}
			result.round(0.0);
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
			addNextSite(S1[l],current_wave_function.get_component(k),k);
			addNextSite(S2[l],current_wave_function.get_component(k),k);

			if (active_site){
				for (size_t r = 0; r < k; ++r){
					coeffs1.push(V[{l,k,r,k}] - V[{k,l,r,k}]);
					coeffs2.push(V[{r,k,l,k}] - V[{r,k,k,l}]);
				}
//				tests1(coeffs1,k,false);
//				tests1(coeffs2,k,true);
				auto tmpk = addNextSiteIter(current_wave_function,T[{l,k}]*e0,k);
				addNextSite(Q1[l][k],e0,k);
				addNextSite(P2[l][k],e0,k);
				S1[l] += sign*tmpk + sign*Q1[l][k] + addNextSiteIter(build1SiteTensor(coeffs1, k, false),e1,k);
				S2[l] += sign* P2[l][k];
				S2[l] += addNextSiteIter(build1SiteTensor(coeffs2, k, true),e1,k);
			} else {
				addNextSite(P1[l][k],e1,k);
				addNextSite(Q2[l][k],e1,k);
				S1[l] += sign*P1[l][k]; //TODO check P1[k][l] or P1[l][k]
				S2[l] += sign*addNextSiteIter(current_wave_function,T[{l,k}]*e1,k) +  sign*Q2[l][k];
			}
			if (k%25 == 0){
				S1[l].round(0.0);
				S2[l].round(0.0);
			}


//			//final sum
//			S[s] = S_tmp + S_tmp2 - S_tmpP + S_tmpQ - tmp;

		}

		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q^* a_p
		 * overwrites Q from the last site
		 */
		void update2SiteOperatorQ(const size_t l, const size_t m, const size_t dim){// TODO check if Q1 and Q2 are definitively different
			Index ii,jj,kk;
			size_t k = dim - 1;
			std::queue<value_t> coeffs1,coeffs2;
			const Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			const Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});
			//first summand, til k - 1
			time_t begin_time = time (NULL);
			addNextSite(Q1[l][m],current_wave_function.get_component(k),k);
			update_Q1ext += time (NULL) -  begin_time;
			begin_time = time (NULL);
			addNextSite(Q2[l][m],current_wave_function.get_component(k),k);
			update_Q2ext += time (NULL) -  begin_time;
			if (active_site){
				for (size_t s = 0; s < k; ++s){
					coeffs1.push(V[{l,s,m,k}] - V[{s,l,m,k}]);
					coeffs2.push(V[{l,s,m,k}] - V[{k,l,m,s}]);
				}
//				tests1(coeffs1,k,true);
//				tests1(coeffs2,k,true);
				value_t val = V[{l,k,m,k}]-V[{k,l,m,k}];
				begin_time = time (NULL);
				Q1[l][m] += addNextSiteIter(current_wave_function,val*e1,k) + sign*addNextSiteIter(build1SiteTensor(coeffs1, dim-1, true),e0,k);
				update_Q1add += time (NULL) -  begin_time;
				begin_time = time (NULL);
				Q2[l][m] += addNextSiteIter(current_wave_function,val*e1,k) + sign*addNextSiteIter(build1SiteTensor(coeffs2, dim-1, true),e0,k);
				update_Q2add += time (NULL) -  begin_time;

			} else {
				for (size_t s = 0; s < k; ++s){
					coeffs1.push(V[{l,k,m,s}] - V[{k,l,m,s}]);
					coeffs2.push(V[{l,k,m,s}] - V[{s,l,m,k}]);
				}
//				tests1(coeffs1,k,false);
//				tests1(coeffs2,k,false);
				begin_time = time (NULL);
				Q1[l][m] -=  sign*addNextSiteIter(build1SiteTensor(coeffs1, dim-1, false),e1,k);
				update_Q1add += time (NULL) -  begin_time;
				begin_time = time (NULL);
				Q2[l][m] -=  sign*addNextSiteIter(build1SiteTensor(coeffs2, dim-1, false),e1,k);
				update_Q2add += time (NULL) -  begin_time;
			}
			if (k%25 == 0){
				Q1[l][m].round(0.0);
				Q2[l][m].round(0.0);
			}

//			//final sum
//			Q[r][s] = Q_tmp +  Q_tmp2 + tmp + tmp_t;
//
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


			//first summand, til k - 1, Note there are no summands for only since they are 0
			addNextSite(P1[l][m],current_wave_function.get_component(k),k);
			addNextSite(P2[l][m],current_wave_function.get_component(k),k);

			if (active_site){
				for (size_t s = 0; s < k; ++s){
					coeffs1.push(V[{l,m,s,k}] - V[{m,l,s,k}]);
				}
//				tests1(coeffs1,k,false);
				P1[l][m] -= sign*addNextSiteIter(build1SiteTensor(coeffs1, dim-1, false),e0,k);
			} else {
				for (size_t s = 0; s < k; ++s){
					coeffs1.push(V[{k,s,l,m}] - V[{k,s,m,l}]);
				}
//				tests1(coeffs1,k,true);
				P2[l][m] -= sign*addNextSiteIter(build1SiteTensor(coeffs1, dim-1, true),e1,k);
			}
			if (k%25 == 0){
				P1[l][m].round(0.0);
				P2[l][m].round(0.0);
			}

//			//final sum
//			P[r][s] = P_tmp + tmp;
		}


		/*
		 * Builds the sum over one index, sum_p w_prqs a_p, as a rank 2 operator
		 */
		TTTensor build1SiteTensor(std::queue<value_t> coeffs, const size_t dim, const bool transpose){
			XERUS_REQUIRE(coeffs.size() <= dim,"Number of coefficients is larger than dimension of result");
			time_t begin_time = time (NULL);
			TTTensor result(std::vector<size_t>(dim,2));
			size_t site = 0;
			Tensor e0 = Tensor::dirac({1,2,1},{0,0,0});
			Tensor e1 = Tensor::dirac({1,2,1},{0,1,0});
			while (!coeffs.empty()){
				value_t coeff = coeffs.front();
				coeffs.pop();
				bool active_s = std::binary_search (current_sample.begin(), current_sample.end(), site);
				if (site == 0){
					if (coeffs.empty()){
						Tensor tmp({1,2,1});
						if (!transpose == active_s)
							tmp.offset_add(transpose ? coeff*e1 : coeff*e0,{0,0,0});
						result.set_component(site,tmp);
					} else {
						Tensor tmp({1,2,2});
						tmp.offset_add(active_s ? (-1)*e1 : e0,{0,0,0});
						if (!transpose == active_s)
							tmp.offset_add(transpose ? coeff*e1 : coeff*e0,{0,0,1});
						result.set_component(site,tmp);
					}
				} else if (coeffs.empty()) {
					Tensor tmp({2,2,1});
					tmp.offset_add(active_s ? e1 : e0,{1,0,0});
					if (!transpose == active_s){
						tmp.offset_add(transpose ? coeff*e1 : coeff*e0,{0,0,0});
					}
					result.set_component(site,tmp);

				} else {
					Tensor tmp({2,2,2});
					tmp.offset_add(active_s ? (-1)*e1 : e0,{0,0,0});
					tmp.offset_add(active_s ? e1 : e0,{1,0,1});
					if (!transpose == active_s){
						tmp.offset_add(transpose ? coeff*e1 : coeff*e0,{0,0,1});
					}
					result.set_component(site,tmp);
				}
				++site;
			}


			//result.round(0.0); //TODO necessary ?
			one_site += time(NULL) - begin_time;
			return result;
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

		void addNextSite(TTTensor &tensor_train, const Tensor next_comp,size_t k){
			time_t begin_time = time (NULL);
			size_t dim = 2;
			if (tensor_train.dimensions.size() == k){
				tensor_train.dimensions.emplace_back(dim);
				tensor_train.externalLinks.emplace_back(k+1, 1, dim, false);
				std::vector<TensorNetwork::Link> neighbors;
				neighbors.emplace_back(k, 2, 1, false);
				neighbors.emplace_back(-1, k, dim, true);
				neighbors.emplace_back(k+2, 0, 1, false);
				auto back = tensor_train.nodes.back();
				back.neighbors[0].other = k+1;
				tensor_train.nodes.pop_back();
				tensor_train.nodes.emplace_back(std::make_unique<Tensor>(next_comp), std::move(neighbors) );
				tensor_train.nodes.emplace_back(back);
				tensor_train.require_correct_format();
			} else
				tensor_train.set_component(k,next_comp);
			copy_tt2 += time(NULL) - begin_time;
		}


		void testResult(const size_t dim){
			Index ii,jj;

			auto brute = build_brute(dim);

			auto unit_tmp = makeUnitVector(current_sample, dim);
			TTTensor result_tmp;
			result_tmp(ii&0) = brute(ii/2,jj/2) * unit_tmp(jj&0);
			XERUS_LOG(info, "Test Result dim = " << dim << " error = " << ((result - result_tmp).frob_norm() < 10e-14 ? 0 : (result - result_tmp).frob_norm()));

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
			XERUS_LOG(info, "Test S1 dim = " << dim << " and l = " << l << "           error = " << ((S1[l] - tmpS1_b).frob_norm() < 10e-14 ? 0 :(S1[l] - tmpS1_b).frob_norm()));
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
			XERUS_LOG(info, "Test S2 dim = " << dim << " and l = " << l << "           error = " << ((S2[l] - tmpS2_b).frob_norm()< 10e-14 ? 0 :(S2[l] - tmpS2_b).frob_norm()));
		}
		void testQ1(const size_t l, size_t m, const size_t dim){
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
			XERUS_LOG(info, "Test Q1 dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << ((Q1[l][m] - tmpQ1_b).frob_norm() < 10e-14 ? 0 :(Q1[l][m] - tmpQ1_b).frob_norm()));
		}

		void testQ2(const size_t l, size_t m, const size_t dim){
			Index ii,jj;
			TTTensor tmpQ2_b(std::vector<size_t>(dim,2));
			TTTensor tt_tmp;
			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t ss = 0; ss < dim;++ss){
				for(size_t pp = 0; pp < dim; ++pp){
					auto tmp = (V[{m,pp,l,ss}] - V[{m,pp,ss,l}])*return_one_e_ac(pp, ss, dim);
					tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
					tmpQ2_b += tt_tmp;
				}
			}
			XERUS_LOG(info, "Test Q2 dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << ((Q2[l][m] - tmpQ2_b).frob_norm()< 10e-14 ? 0 :(Q2[l][m] - tmpQ2_b).frob_norm()));
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
			XERUS_LOG(info, "Test P1 dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << ((P1[l][m] - tmpP1_b).frob_norm()< 10e-14 ? 0 :(P1[l][m] - tmpP1_b).frob_norm()));
		}

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
			XERUS_LOG(info, "Test P2 dim = " << dim << " and l = " << l <<  " and m = " << m <<" error = " << ((P2[l][m] - tmpP2_b).frob_norm()< 10e-14 ? 0 :(P2[l][m] - tmpP2_b).frob_norm()));
		}

		void tests1(std::queue<value_t> coeffs, size_t dim, bool transpose){
			Index ii,jj;
			TTTensor tmps1_b(std::vector<size_t>(dim,2));
			TTTensor tt_tmp;
			auto build1s = build1SiteTensor(coeffs, dim, transpose);

			auto unit_tmp = makeUnitVector(current_sample, dim);
			for(size_t rr =0; rr < dim;++rr){
				value_t coeff = coeffs.front();
				coeffs.pop();
				TTOperator tmp;
				if (transpose)
					tmp = coeff*return_create(rr, dim);
				else
					tmp = coeff*return_annil(rr, dim);
				tt_tmp(ii&0) = tmp(ii/2,jj/2) * unit_tmp(jj&0);
				tmps1_b += tt_tmp;


			}

			XERUS_LOG(info, "Test s1 dim = " << dim << "                     error = " << ((build1s - tmps1_b).frob_norm()< 10e-14 ? 0 : (build1s - tmps1_b).frob_norm()));
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


