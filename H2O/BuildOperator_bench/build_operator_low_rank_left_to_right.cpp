#include <xerus.h>

#include <queue>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>


#define debug 0

using namespace xerus;
using xerus::misc::operator<<;

xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);
xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac_partial(size_t i, size_t j, size_t k, size_t d);
xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);



class BuildingOperatorL2R{
	public:
		size_t d;
		Tensor V;
		Tensor T;
		Tensor N;
		std::string path;
		std::vector<std::vector<TTOperator>> P;
		std::vector<std::vector<TTOperator>> Q;
		std::vector<TTOperator> S;
		TTOperator H;

		/*
		 * Constructor
		 */
		BuildingOperatorL2R(size_t _d,std::string _path)
		: d(_d), path(_path){
			XERUS_LOG(BuildOpL2R, "---- Initializing Building Operator L2R ----");
			XERUS_LOG(BuildOpL2R, "- Loading 1e and 2e Integrals and Nuclear Potential");
			T = load_1e_int();
			V = load_2e_int();
			N = load_nuclear();
			XERUS_LOG(BuildOpL2R, "- Initializing Storage Operators H,S,Q,P");
			ini_4_site_operator_H();
			ini_4_site_operator_S();
			ini_4_site_operator_Q();
			ini_4_site_operator_P();
		}

		/*
		 * Build Operator
		 */
		void build(){
			XERUS_LOG(BuildOpL2R, "---- Building Operator ----");
			for (size_t k = 1; k < d; ++k){
				step(k);
				XERUS_LOG(BuildOpL2R, "- Ranks of Operator " << H.ranks());
			}
		}

		/*
		 * One step from k to k + 1
		 */
		void step(const size_t k){
			XERUS_LOG(BuildOpL2R, "- Step " << k);
			update_4_site_operator_H(k+1);

			for (size_t s = k; s < d; ++s)
				update_3_site_operator_S(s,k+1);

			for (size_t r = k; r < d; ++r){
				for (size_t s = k; s < d; ++s){
					update_2_site_operator_Q(r,s,k+1);
					update_2_site_operator_P(r,s,k+1);
				}
			}


		}

		/*
		 * initializes operator H (k = 0)
		 */
		void ini_4_site_operator_H(){
			H = TTOperator({2,2});
			auto tmp = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			tmp[{0,1,1,0}] = T[{0,0}];
			H.set_component(0,tmp);
		}

		/*
		 * initializes operator S (k = 0)
		 */
		void ini_4_site_operator_S(){
			for(size_t s = 0; s < d; ++s){
				S.emplace_back(TTOperator({2,2}));
				auto tmp = xerus::Tensor({1,2,2,1});
				tmp[{0,1,0,0}] = T[{0,s}];
				S[s].set_component(0,tmp);
			}
		}

		/*
		 * initializes operator S (k = 0)
		 */
		void ini_4_site_operator_Q(){
			for (size_t r = 0; r < d; ++r){
				std::vector<TTOperator> tmp2;
				Q.emplace_back(tmp2);
				for(size_t s = 0; s < d; ++s){
					Q[r].emplace_back(TTOperator({2,2}));
					auto tmp = xerus::Tensor({1,2,2,1});
					tmp[{0,1,1,0}]  = V[{0,r,0,s}] - V[{r,0,0,s}];
					Q[r][s].set_component(0,tmp);
				}
			}
		}

		/*
		 * initializes operator S (k = 0)
		 */
		void ini_4_site_operator_P(){
			for (size_t r = 0; r < d; ++r){
				std::vector<TTOperator> tmp2;
				P.emplace_back(tmp2);
				for(size_t s = 0; s < d; ++s){
					P[r].emplace_back(TTOperator({2,2}));
				}
			}
		}

		/*
		 * Builds the hamiltonian from left to right, uses S and Q
		 * overwrites H from the last site
		 */
		void update_4_site_operator_H(const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;
			XERUS_LOG(info,"k = " << k);

			//first summand
			TTOperator H_tmp = add_identity(H);

			//second summand only k
			TTOperator H_tmp2 = TTOperator(std::vector<size_t>(2*dim,2));
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			for (size_t i = 0; i < k; ++i)
				H_tmp2.set_component(i,id);
			value_t coeff = T[{k,k}];
			auto aa = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			aa[{0,1,1,0}] = coeff;
			H_tmp2.set_component(k,aa);

			//third summand
			TTOperator H_tmpS,H_tmpS_t,annil;
			H_tmpS = add_identity(S[k]);
			annil = return_annil(k,dim);
			H_tmpS(ii/2,jj/2) =  H_tmpS(ii/2,kk/2)*annil(kk/2,jj/2);
			H_tmpS_t(ii/2,jj/2) = H_tmpS(jj/2,ii/2); // transpose

			//fourth summand
			TTOperator H_tmpQ;
			H_tmpQ = add_identity(Q[k][k]);
			auto particle = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			particle[{0,1,1,0}] = 1.0;
			H_tmpQ.set_component(k,particle);

#if debug
			if(dim == 4){
				auto H_tmp_brute = add_identity(build_brute(3));
				auto tmp = H_tmp_brute - H_tmp;
				tmp.round(0.0);
				auto H_tmp2_brute = T[{k,k}] * return_one_e_ac(k,k, dim);
				auto tmp2 = H_tmp2_brute - H_tmp2;
				tmp2.round(0.0);


				auto H_tmpQ_brute = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <3; ++p){
					for(size_t q = 0; q <3; ++q){
						H_tmpQ_brute+=0.5*V[{p,k,q,k}]*return_two_e_ac(p,k,q,k, dim);
						H_tmpQ_brute+=0.5*V[{k,p,q,k}]*return_two_e_ac(k,p,q,k, dim);
						H_tmpQ_brute+=0.5*V[{p,k,k,q}]*return_two_e_ac(p,k,k,q, dim);
						H_tmpQ_brute+=0.5*V[{k,p,k,q}]*return_two_e_ac(k,p,k,q, dim);
					}
				}
				auto tmpQ = H_tmpQ_brute + H_tmpQ;
				tmpQ.round(0.0);

				auto H_tmpS_brute = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <3; ++p){
					H_tmpS_brute+=T[{p,k}]*return_one_e_ac(p,k, dim);
					H_tmpS_brute+=T[{k,p}]*return_one_e_ac(k,p, dim);
					for(size_t q = 0; q <3; ++q){
						for(size_t r = 0; r <3; ++r){
							H_tmpS_brute+=0.5*V[{p,q,r,k}]*return_two_e_ac(p,q,r,k, dim);
							H_tmpS_brute+=0.5*V[{p,q,k,r}]*return_two_e_ac(p,q,k,r, dim);
							H_tmpS_brute+=0.5*V[{p,k,q,r}]*return_two_e_ac(p,k,q,r, dim);
							H_tmpS_brute+=0.5*V[{k,p,q,r}]*return_two_e_ac(k,p,q,r, dim);
						}
					}
				}

				auto H_tmpS_brute2 = TTOperator(std::vector<size_t>(2*dim,2));
				auto H_tmpS_brute2_t = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <3; ++p){
					H_tmpS_brute2+=T[{p,k}]*return_one_e_ac(p,k, dim);
					H_tmpS_brute2_t+=T[{k,p}]*return_one_e_ac(k,p, dim);
					for(size_t q = 0; q <3; ++q){
						for(size_t r = 0; r <3; ++r){
							H_tmpS_brute2+=0.5*(V[{p,q,r,k}]-V[{p,q,k,r}])*return_two_e_ac(p,q,r,k, dim);
							H_tmpS_brute2_t+=0.5*(V[{k,p,q,r}]-V[{p,k,q,r}])*return_two_e_ac(k,p,q,r, dim);
						}
					}
				}
				auto tmpS = H_tmpS_brute - H_tmpS - H_tmpS_t;
				tmpS.round(0.0);
				auto tmpS2 = H_tmpS_brute - H_tmpS_brute2 - H_tmpS_brute2_t;
				tmpS2.round(0.0);
				auto tmpS3 = H_tmpS - H_tmpS_brute2;
				tmpS3.round(0.0);
				auto tmpS4 = H_tmpS_t - H_tmpS_brute2_t;
				tmpS4.round(0.0);


				XERUS_LOG(info, "1st Summand (HxI)  " << tmp.frob_norm());
				XERUS_LOG(info, "2nd Summand (IxH)  " << tmp2.frob_norm());
				XERUS_LOG(info, "3rd Summand (Q)    " << tmpQ.frob_norm());
				XERUS_LOG(info, "4th Summand1(S+St) " << tmpS.frob_norm());
				XERUS_LOG(info, "4th Summand2(S+St) " << tmpS2.frob_norm());
				XERUS_LOG(info, "4th Summand3(S+St) " << tmpS3.frob_norm());
				XERUS_LOG(info, "4th Summand4(S+St) " << tmpS4.frob_norm());

			}
#endif
			H = H_tmp + H_tmp2 + H_tmpS + H_tmpS_t + H_tmpQ;
			H.round(0.0);
		}

		/*
		 * Builds the sum over three indices, sum_pqr w_prqs a_p^* a_q^* a_r and includes 1e parts
		 * overwrites S from the last site
		 */
		void update_3_site_operator_S(const size_t s, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;

			//first summand, til k - 1
			TTOperator S_tmp = add_identity(S[s]);

			//second summand only k, i.e. only 1e part
			TTOperator S_tmp2 = T[{k,s}]*return_create(k, dim);

			//third summand
			TTOperator S_tmpP,annil;
			S_tmpP = add_identity(P[k][s]);
			annil = return_annil(k,dim);
			S_tmpP(ii/2,jj/2) =  S_tmpP(ii/2,kk/2)*annil(kk/2,jj/2);

			//fourth summand
			TTOperator S_tmpQ,create;
			S_tmpQ = add_identity(Q[k][s]);
			create = return_create(k,dim);
			S_tmpQ(ii/2,jj/2) = create(ii/2,kk/2) *  S_tmpQ(kk/2,jj/2);

			//fifth summand
			std::queue<value_t> coeffs;
			for (size_t p = 0; p < k; ++p){
				auto coeff = V[{p,k,k,s}] - V[{p,k,s,k}];
				coeffs.push(coeff);
			}
			TTOperator tmp = build_1_site_operator(coeffs, dim, true);
			auto aa = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			aa[{0,1,1,0}] = 1.0;
			tmp.set_component(k,aa);
			tmp.require_correct_format();

#if debug
			//if(dim == 4){
				auto S_tmpP_brute = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <k; ++p){
					for(size_t q = 0; q <k; ++q){
						S_tmpP_brute+=0.5*(V[{p,q,k,s}] - V[{p,q,s,k}])*return_two_e_ac_partial(p,q,k, dim);
					}
				}
				auto tmpP = S_tmpP_brute - S_tmpP;
				tmpP.round(0.0);

				auto S_tmpQ_brute = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <k; ++p){
					for(size_t q = 0; q <k; ++q){
						S_tmpQ_brute+=0.5*(V[{p,k,q,s}] - V[{p,k,s,q}])*return_two_e_ac_partial(p,k,q, dim);
						S_tmpQ_brute+=0.5*(V[{k,p,q,s}] - V[{k,p,s,q}])*return_two_e_ac_partial(k,p,q, dim);
					}
				}
				auto tmpQ = S_tmpQ_brute + S_tmpQ;
				tmpQ.round(0.0);

				auto S_tmpR_brute = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <k; ++p){
					S_tmpR_brute+=0.5*(V[{p,k,k,s}] - V[{p,k,s,k}])*return_two_e_ac_partial(p,k,k, dim);
					S_tmpR_brute+=0.5*(V[{k,p,k,s}] - V[{k,p,s,k}])*return_two_e_ac_partial(k,p,k, dim);
				}
				auto tmpR = S_tmpR_brute - tmp;
				tmpR.round(0.0);
				auto S_tmp2_brute = TTOperator(std::vector<size_t>(2*dim,2));
				S_tmp2_brute+=T[{k,s}]*return_create(k, dim);
				auto tmp2 = S_tmp2_brute - S_tmp2;
				tmp2.round(0.0);


				XERUS_LOG(info, "1st Summand (PxI)  " << tmpP.frob_norm());
				XERUS_LOG(info, "2nd Summand (QxI)  " << tmpQ.frob_norm());
				XERUS_LOG(info, "3nd Summand (RxI)  " << tmpR.frob_norm());
				XERUS_LOG(info, "4th Summand (IxS)  " << tmp2.frob_norm());
			//}
				auto S_tmp_brute = TTOperator(std::vector<size_t>(2*dim,2));
				for(size_t p = 0; p <dim; ++p){
					S_tmp_brute+=T[{p,s}]*return_create(p, dim);
					for(size_t q = 0; q <dim; ++q){
						for(size_t r = 0; r <dim; ++r){
							S_tmp_brute+=0.5*(V[{p,q,r,s}] - V[{p,q,s,r}])*return_two_e_ac_partial(p,q,r, dim);
						}
					}
				}
				auto tmpG = S_tmp + S_tmp2 + S_tmpP - S_tmpQ + tmp - S_tmp_brute;
				tmpG.round(0.0);
				XERUS_LOG(info, "5th Summand (S)    " << tmpG.frob_norm());
#endif

			//final sum
			S[s] = S_tmp + S_tmp2 - S_tmpP + S_tmpQ - tmp;
			S[s].round(0.0);

		}



		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q^* a_p
		 * overwrites Q from the last site
		 */
		void update_2_site_operator_Q(const size_t r, const size_t s, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;

			//first summand, til k - 1
			TTOperator Q_tmp = add_identity(Q[r][s]);

			//second summand only k
			TTOperator Q_tmp2 = TTOperator(std::vector<size_t>(2*dim,2));
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			for (size_t i = 0; i < k; ++i)
				Q_tmp2.set_component(i,id);
			value_t coeff = V[{k,r,k,s}] - V[{r,k,k,s}];
			auto aa = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			aa[{0,1,1,0}] = coeff;
			Q_tmp2.set_component(k,aa);

			//third summand interaction terms
			TTOperator tmp,tmp_t, create,annil;
			std::queue<value_t> coeffs,coeffs2;
			for (size_t p = 0; p < k; ++p){
				coeff = V[{p,r,k,s}] - V[{r,p,k,s}];
				coeffs.push(coeff);
				coeff = V[{k,r,p,s}] - V[{r,k,p,s}];
				coeffs2.push(coeff);
			}
			tmp = build_1_site_operator(coeffs, dim,true);
			annil = return_annil(k,dim);
			tmp(ii/2,jj/2) = tmp(ii/2,kk/2) * annil(kk/2,jj/2);

			//fourth summand interaction terms
			tmp_t = build_1_site_operator(coeffs2, dim);
			create = return_create(k,dim);
			tmp_t(ii/2,jj/2) = create(ii/2,kk/2) * tmp_t(kk/2,jj/2);

			//final sum
			Q[r][s] = Q_tmp +  Q_tmp2 + tmp + tmp_t;
			Q[r][s].round(0.0);
		}


		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q a_p
		 * overwrites P from the last site
		 */
		void update_2_site_operator_P(const size_t r, const size_t s, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;

			//first summand, til k - 1, Note there are no summands for only since they are 0
			TTOperator P_tmp = add_identity(P[r][s]);

			//interaction terms
			std::queue<value_t> coeffs;
			for (size_t p = 0; p < k; ++p){
				value_t coeff = V[{p,k,r,s}] - V[{p,k,s,r}];
				coeffs.push(coeff);
			}
			TTOperator tmp = build_1_site_operator(coeffs, dim, true);
			TTOperator create = return_create(k,dim);
			tmp(ii/2,jj/2) =  tmp(ii/2,kk/2) * create(kk/2,jj/2);

			//final sum
			P[r][s] = P_tmp + tmp;
			P[r][s].round(0.0);
		}


		/*
		 * Builds the sum over one index, sum_p w_prqs a_p, as a rank 2 operator
		 */
		TTOperator build_1_site_operator(std::queue<value_t> coeffs, const size_t dim, const bool transpose = false){
			XERUS_REQUIRE(coeffs.size() < dim,"Number of coefficients is larger than dimension of result");
			TTOperator result(std::vector<size_t>(2*dim,2));
			size_t comp = 0;
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			auto s = xerus::Tensor::identity({2,2});
			s.reinterpret_dimensions({1,2,2,1});
			s[{0,1,1,0}] = -1.0;
			auto a = xerus::Tensor({1,2,2,1});
			if (transpose)
				a[{0,1,0,0}] = 1.0;
			else
				a[{0,0,1,0}] = 1.0;
			while (!coeffs.empty()){ // set the rank 2 blocks
				value_t coeff = coeffs.front();
				coeffs.pop();
				if (comp == 0){
					if (coeffs.empty()){
						Tensor tmp = Tensor({1,2,2,1});
						tmp.offset_add(coeff*a,{0,0,0,0});
						result.set_component(comp,tmp);
					} else {
						Tensor tmp = Tensor({1,2,2,2});
						tmp.offset_add(s,{0,0,0,0});
						tmp.offset_add(coeff*a,{0,0,0,1});
						result.set_component(comp,tmp);
					}
				} else if (coeffs.empty()){
					Tensor tmp = Tensor({2,2,2,1});
					tmp.offset_add(coeff*a,{0,0,0,0});
					tmp.offset_add(id,{1,0,0,0});
					result.set_component(comp,tmp);
				} else {
					Tensor tmp = Tensor({2,2,2,2});
					tmp.offset_add(s,{0,0,0,0});
					tmp.offset_add(coeff*a,{0,0,0,1});
					tmp.offset_add(id,{1,0,0,1});
					result.set_component(comp,tmp);
				}
				++comp;
			}
			while(comp < dim){ // the rest are identities
				result.set_component(comp,id);
				++comp;
			}
			return result;
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
		 * Annihilation Operator with operator at position i
		 */
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

		/*
		 * Creation Operator with operator at position i
		 */
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

		/*
		 * Loads the nuclear Potential
		 */
		xerus::Tensor load_nuclear(){
				auto Nuc = xerus::Tensor({1});
				std::string line;
				std::ifstream input;
				input.open (path);
				size_t count = 0;
				while ( std::getline (input,line) )
				{
					count++;
					if (count > 4){
						std::vector<std::string> l;
						boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
						if (std::stoi(l[1]) == 0 && std::stoi(l[3]) == 0){
							Nuc[{0}] = stod(l[0]);
						}
					}
				}
				input.close();
				return Nuc;
			}

		/*
		 * Loads the 1 electron integral
		 */
		xerus::Tensor load_1e_int(){
				auto nob = d/2;
				auto H = xerus::Tensor({2*nob,2*nob});
				auto H_tmp = xerus::Tensor({nob,nob});
				std::string line;
				std::ifstream input;
				input.open (path);
				size_t count = 0;
				while ( std::getline (input,line) )
				{
					count++;
					if (count > 4){
						std::vector<std::string> l;
						boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
						if (std::stoi(l[1]) != 0 && std::stoi(l[3]) == 0){
							H_tmp[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1}] = stod(l[0]);
						}
					}
				}
				input.close();
				for (size_t i = 0; i < nob; i++){
					for (size_t j = 0; j < nob; j++){
						auto val = H_tmp[{i,j}];
						H[{2*i,2*j}] = val;
						H[{2*j,2*i}] = val;
						H[{2*i+1,2*j+1}] = val;
						H[{2*j+1,2*i+1}] = val;
					}
				}
				return H;
			}

		/*
		 * Loads the 2 electron integral
		 */
		xerus::Tensor load_2e_int(){
				auto nob = d/2;
				auto V = xerus::Tensor({2*nob,2*nob,2*nob,2*nob});
				auto V_tmp = xerus::Tensor({nob,nob,nob,nob});
				auto V_tmp2 = xerus::Tensor({nob,nob,nob,nob});
				std::string line;
				std::ifstream input;
				input.open (path);
				size_t count = 0;
				while ( std::getline (input,line) )
				{
					count++;
					if (count > 4){
						std::vector<std::string> l;
						boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
						if (std::stoi(l[1]) != 0 && std::stoi(l[3]) != 0){
							V_tmp[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1,static_cast<size_t>(std::stoi(l[3]))-1,static_cast<size_t>(std::stoi(l[4]))-1}] = stod(l[0]);
						}
					}
				}
				input.close();
				for (size_t i = 0; i < nob; i++){
						for (size_t j = 0; j <= i; j++){
							for (size_t k = 0; k<= i; k++){
								for (size_t l = 0; l <= (i==k ? j : k); l++){
									auto value = V_tmp[{i,j,k,l}];
									V_tmp2[{i,k,j,l}] = value;
									V_tmp2[{j,k,i,l}] = value;
									V_tmp2[{i,l,j,k}] = value;
									V_tmp2[{j,l,i,k}] = value;
									V_tmp2[{k,i,l,j}] = value;
									V_tmp2[{l,i,k,j}] = value;
									V_tmp2[{k,j,l,i}] = value;
									V_tmp2[{l,j,k,i}] = value;
								}
							}
						}
				}
				for (size_t i = 0; i < nob; i++){
							for (size_t j = 0; j < nob; j++){
								for (size_t k = 0; k < nob; k++){
									for (size_t l = 0; l < nob; l++){
										auto value = V_tmp2[{i,j,k,l}];
										V[{2*i,2*j,2*k,2*l}] = value;
										V[{2*i+1,2*j,2*k+1,2*l}] = value;
										V[{2*i,2*j+1,2*k,2*l+1}] = value;
										V[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
									}
								}
							}
				}

				return V;
			}
		TTOperator build_brute(const size_t dim){
			auto opH = xerus::TTOperator(std::vector<size_t>(2*dim,2));
			XERUS_LOG(info,"dim = " << dim);
			for (size_t i = 0; i < dim; i++){
					for (size_t j = 0; j < dim; j++){
							value_t val = T[{i , j}];
							opH += val * return_one_e_ac(i,j,dim);
					}
			}

			auto opV = xerus::TTOperator(std::vector<size_t>(2*dim,2));
			for (size_t i = 0; i < dim; i++){
				for (size_t j = 0; j < dim; j++){
					for (size_t k = 0; k < dim; k++){
						for (size_t l = 0; l < dim; l++){
							value_t val = V[{i,j,k,l}];
							if (std::abs(val) < 10e-14 || i == j || k == l )//|| a == b || c: == d) // TODO check i == j || k == l
								continue;
							opV += 0.5*val * return_two_e_ac(i,j,k,l,dim); // TODO: Note change back swap k,l
						}
					}
				}
			}
			auto res = opH + opV;
			res.round(0.0);
			return res;
		}
};


int main(){
	XERUS_LOG(info, "---- Start building operator left to right! ----");

	size_t d = 48;
	std::string path = "../FCIDUMP.h2o_24";
	BuildingOperatorL2R builder(d,path);

	std::string name = "../data/T_H2O_48_bench.tensor";
	std::ofstream write(name.c_str());
	misc::stream_writer(write,builder.T,xerus::misc::FileFormat::BINARY);
	write.close();

	name = "../data/V_H2O_48_bench.tensor";
	std::ofstream write2(name.c_str());
	misc::stream_writer(write2,builder.V,xerus::misc::FileFormat::BINARY);
	write2.close();

	return 0;

	XERUS_LOG(test,"Testing build");
	builder.build();


	XERUS_LOG(info, "---- Save Test Result ----");
	name = "../hamiltonian_H2O_48_full_3.ttoperator";
	std::ofstream write3(name.c_str());
	misc::stream_writer(write3,builder.H,xerus::misc::FileFormat::BINARY);
	write3.close();


	XERUS_LOG(test,"Testing correctnes of operator comparuing with brute force ");
	//auto brute = builder.build_brute(10);
	//XERUS_LOG(test, brute.ranks());
	XERUS_LOG(test, builder.H.ranks());


	return 0;
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

xerus::TTOperator return_two_e_ac_partial(size_t i, size_t j, size_t k, size_t d){ //todo test
	auto cr1 = return_create(i,d);
	auto cr2 = return_create(j,d);
	auto an1 = return_annil(k,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk,ll;
	res(ii/2,ll/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) ;
	return res;
}

xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim){
	auto res = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	if (a!=b && c!=d)
		res += return_two_e_ac(a,b,d,c,dim);
		res += return_two_e_ac(a+1,b+1,d+1,c+1,dim);
	if (c != d)
		res += return_two_e_ac(a+1,b,d,c+1,dim);
	res += return_two_e_ac(a,b+1,d+1,c,dim);


	return res;
}

