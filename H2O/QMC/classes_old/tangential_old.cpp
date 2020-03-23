#include <xerus.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <stdlib.h>

#include "../classes_old/contractpsihek.cpp"
#include "../classes_old/metropolis.cpp"
#include "../classes_old/probabilityfunctions.cpp"
#include "../classes_old/trialfunctions.cpp"
#include "../classes_old/unitvectorprojection.cpp"
#include "../GradientMethods/basic.cpp"
#include "../loading_tensors.cpp"



class Tangential{
	public:
		size_t d,p, iterations;
		std::string path_T, path_V;
		value_t shift;
		std::vector<size_t> start_sample;
		TTTensor phi;
		ContractPsiHek builder;

		Tangential(size_t _d, size_t _p, size_t _iter, std::string _path_T, std::string _path_V, value_t _shift, std::vector<size_t> _s, TTTensor _phi) \
				: d(_d), p(_p), iterations(_iter), path_T(_path_T), path_V(_path_V), shift(_shift), start_sample(_s), phi(_phi), builder(phi,d,p,path_T,path_V,0.0, shift) {
		}

		void update(TTTensor _phi){
			phi = _phi;
		}

		value_t get_eigenvalue(){
			value_t ev_exact,res,psi_ek,factor;
			XERUS_LOG(info,"Calculate Raleigh quotient");
			Tensor test_component = get_test_component(0,phi,ev_exact, true);
			PsiProbabilityFunction PsiPF(phi);
			Metropolis<PsiProbabilityFunction> markow1(&PsiPF, TrialSample, start_sample, d);
			std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap1;
			runMetropolis<PsiProbabilityFunction>(&markow1,umap1,iterations);

			value_t ev = 0;
			//Eigenvalue
			for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap1) {
				builder.reset(pair.first);
				res = builder.contract();
				psi_ek = PsiPF.values[pair.first];
				factor = res* (value_t) pair.second.first/psi_ek;
				ev += factor;
				//XERUS_LOG(info, "{" << pair.first << ": " << std::setprecision(3) << pair.second  << " " << res << "}");
			}
			ev /= (value_t) iterations;

			XERUS_LOG(info, "Number of samples umap1: " << umap1.size());
			XERUS_LOG(info, "Exact ev    " << ev_exact);
			XERUS_LOG(info,"Ev           "<<ev);
			XERUS_LOG(info,"Ev error     "<<std::abs(ev - ev_exact));
			XERUS_LOG(info,"Calculate Components of tangential vector");
			return ev;
		}

		std::vector<Tensor> get_tangential_components(value_t ev){
			value_t ev_exact,res,psi_ek,factor;
			std::vector<Tensor> results;
			Tensor test_component = get_test_component(0,phi,ev_exact, true);


			for (size_t pos = 5; pos < 6; ++pos){
				value_t prob = 0;
				Tensor test_component = get_test_component(pos,phi,ev_exact, true);
				ProjectorProbabilityFunction PPF(phi,pos, true);
				unitVectorProjection uvP(phi,pos);
				XERUS_LOG(info, "Position = "  << pos);
				XERUS_LOG(info, "P (" << start_sample << ") = " << PPF.P(start_sample));
				XERUS_LOG(info, "Run Metropolis, start sample: " << start_sample);
				Metropolis<ProjectorProbabilityFunction> markow2(&PPF, TrialSample2, start_sample, d);

				std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap2;
				runMetropolis<ProjectorProbabilityFunction>(&markow2,umap2,iterations);


				XERUS_LOG(info,"Caluclate expectation of gradient");
				size_t count = 0;
				value_t sum = 0;
				Tensor result(test_component.dimensions);
				for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap2) {
					builder.reset(pair.first); //setting builder to newest sample!! Important
					res = builder.contract();
					psi_ek = PPF.values[pair.first];
					sum +=psi_ek*psi_ek;
					auto loc_grad = uvP.localProduct(pair.first,true);

					auto idx = makeIndex(pair.first);
					factor = (res - ev*phi[idx])* (value_t) pair.second.first/(psi_ek*psi_ek);
					result += factor * loc_grad;
					prob +=pair.second.second; //TODO check this

					//Debugging
//					count++;
//					if (count % 20 == 0)
//						XERUS_LOG(info, count);
//					if ( pair.second.first > iterations/10000)
//						XERUS_LOG(info, "{" << pair.first << ": " << std::setprecision(3) << pair.second  << " " << res << "}");
				}
				result /= (value_t) iterations;
        Tensor nextComp = result*prob;
				XERUS_LOG(info, "Number of samples umap2: " << umap2.size());

				XERUS_LOG(info,"Sum psi_ek   "<<sum);
				XERUS_LOG(info,"prob         "<< prob);
				XERUS_LOG(info,"test component: "<< test_component.frob_norm() << "\n" << test_component);
				XERUS_LOG(info,"result*prob:    "<< (nextComp).frob_norm() << "\n" << nextComp);
				XERUS_LOG(info,"test component - result*prob (abs):    "<< (test_component-nextComp).frob_norm() );
				XERUS_LOG(info,"test component - result*prob (rel):    "<< (test_component-nextComp).frob_norm() / test_component.frob_norm()<< "\n" << test_component-nextComp);
				results.emplace_back(nextComp);
			}
		return results;
	}

	std::vector<Tensor> get_tangential_components2(value_t ev, value_t accuracy=0.0001){
				value_t ev_exact,res,psi_ek,factor;
				std::vector<Tensor> results;
				Tensor test_component = get_test_component(0,phi,ev_exact, true);


				for (size_t pos = 0; pos < d; ++pos){
					value_t prob = 0, prob2  = 0;
					Tensor test_component = get_test_component(pos,phi,ev_exact, true);
					ProjectorProbabilityFunction PPF(phi,pos, true);
					unitVectorProjection uvP(phi,pos);
					XERUS_LOG(info, "Position = "  << pos);
					XERUS_LOG(info, "P (" << start_sample << ") = " << PPF.P(start_sample));
					XERUS_LOG(info, "Run Metropolis, start sample: " << start_sample);
					Metropolis<ProjectorProbabilityFunction> markow2(&PPF, TrialSample2, start_sample, d);

					std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap2;
					runMetropolis<ProjectorProbabilityFunction>(&markow2,umap2,iterations);


					XERUS_LOG(info,"Caluclate expectation of gradient");
					size_t count = 0;
					Tensor result(test_component.dimensions);
					for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap2) {
						prob +=pair.second.second; //TODO check this
						if ((value_t) pair.second.first > accuracy * (value_t) iterations){
							prob2 +=pair.second.second; //TODO check this
							count++;
							builder.reset(pair.first); //setting builder to newest sample!! Important
							res = builder.contract();
							auto loc_grad = uvP.localProduct(pair.first,true);

							auto idx = makeIndex(pair.first);
							factor = (res - ev*phi[idx]);
							result += factor * loc_grad;
						}
					}
					Tensor nextComp = result;
					XERUS_LOG(info, "Number of samples umap2: " << umap2.size());
					XERUS_LOG(info, "Number of calculations:  " << count);
					XERUS_LOG(info,"prob         "<< prob);
					XERUS_LOG(info,"prob2         "<< prob2);

					XERUS_LOG(info,"test component: "<< test_component.frob_norm() << "\n" << test_component);
					XERUS_LOG(info,"result*prob:    "<< (nextComp).frob_norm() << "\n" << nextComp);
					XERUS_LOG(info,"test component - result*prob (abs):    "<< (test_component-nextComp).frob_norm() );
					XERUS_LOG(info,"test component - result*prob (rel):    "<< (test_component-nextComp).frob_norm() / test_component.frob_norm()<< "\n" << test_component-nextComp);
					results.emplace_back(nextComp);
				}
			return results;
		}


	private:


		template<class ProbabilityFunction>
		void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations){
			std::vector<size_t> next_sample;
			XERUS_LOG(info, "- Build MC Chain -");
			for (size_t i = 0; i < iterations/10; ++i)
				next_sample = markow->getNextSample();
			XERUS_LOG(info, "Start" << next_sample);

			for (size_t i = 0; i < iterations; ++i){
				next_sample = markow->getNextSample();
				auto itr = umap.find(next_sample);
				if (itr == umap.end()){
					umap[next_sample].first = 1;
					umap[next_sample].second = markow->P->P(next_sample);
				} else
					umap[next_sample].first += 1;
				if (i % (iterations / 10) == 0)
					XERUS_LOG(info, i);
			}
		}


		Tensor get_test_component(size_t pos, TTTensor phi, value_t& ev,bool proj){
			TTOperator Hs;
			std::vector<Tensor> tang;
			value_t xx,xHx;


			XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
			read_from_disc("../data/hamiltonian_H2O_48_full_shifted_benchmark.ttoperator",Hs);


			xx = phi.frob_norm();
			phi /= xx; //normalize
			xHx = contract_TT(Hs,phi,phi);
			ev = xHx;
			TangentialOperation top(phi);
			TTOperator id = TTOperator::identity(std::vector<size_t>(2*phi.order(),2));
			tang = top.localProduct(Hs,id,xHx,proj);
			return tang[pos];
		}


		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			return index;
		}


};











