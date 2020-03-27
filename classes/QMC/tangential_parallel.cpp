#include <xerus.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <stdlib.h>
#include <omp.h>

#include "metropolis.cpp"
#include "contractpsihek.cpp"
#include "trialfunctions.cpp"
#include "probabilityfunctions.cpp"
#include "unitvectorprojection.cpp"
#include "../../classes/loading_tensors.cpp"



class Tangential{
	public:
		size_t d,p, iterations;
		std::string path_T, path_V;
		value_t shift;
		std::vector<size_t> start_sample;
		TTTensor phi;
		ContractPsiHek builder;
		std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> eHxValues;
		unitVectorProjection uvP;
		Tangential(size_t _d, size_t _p, size_t _iter, std::string _path_T, std::string _path_V, value_t _shift, std::vector<size_t> _s, TTTensor _phi) \
				: d(_d), p(_p), iterations(_iter), path_T(_path_T), path_V(_path_V), shift(_shift), start_sample(_s), \
					phi(_phi), builder(phi,d,p,path_T,path_V,0.0, shift), uvP(_phi) {
		}

		void update(TTTensor _phi){
			phi = _phi;
			uvP.update(_phi);
			eHxValues.clear();
		}

		value_t get_eigenvalue(){
			value_t ev_exact,res,psi_ek,factor,dk;
			size_t iter_factor = 10;
			PsiProbabilityFunction PsiPF(phi);
			Metropolis<PsiProbabilityFunction> markow1(&PsiPF, TrialSample, start_sample, d);
			std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples;
			runMetropolis<PsiProbabilityFunction>(&markow1,samples,iter_factor*iterations);

			auto samples_keys = extract_keys(samples,0.0);
#pragma omp parallel for schedule(dynamic) shared(eHxValues) firstprivate(builder)
			for (size_t i = 0; i < samples_keys.size(); ++i){
				auto itr = eHxValues.find(samples_keys[i]);
				if (itr == eHxValues.end()){
					 //setting builder to newest sample!! Important
					builder.reset(samples_keys[i]);
					value_t tmp = builder.contract();
#pragma omp critical
								eHxValues[samples_keys[i]] = tmp;
					}
				}
			value_t ev = 0;
			XERUS_LOG(info, "Number of samples for Eigenvalue " << samples.size());
			for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: samples) {
				psi_ek = PsiPF.values[pair.first];
				factor = eHxValues[pair.first]* (value_t) pair.second.first/psi_ek;
				ev += factor;
			}
			ev /= (value_t) iter_factor*iterations;
			return ev;
		}


	std::vector<Tensor> get_tangential_components(value_t ev, value_t accuracy=0.0001){
		value_t ev_exact,res,psi_ek,factor,dk;
		std::vector<Tensor> results; // Vector of Tangential space components

		// Loop for projection of each component
		for (size_t pos = 0; pos < d; ++pos){
			XERUS_LOG(info,"Pos = " << pos);
			std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples;
			XERUS_LOG(info,"Start metropolis");
#pragma omp parallel shared(samples,iterations)
			{
			ProjectorProbabilityFunction2 PPF(phi,pos, true,builder);
			Metropolis<ProjectorProbabilityFunction2> markow2(&PPF, /*TODO check*/ TrialSample2, start_sample, d);
			std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> private_samples;
			runMetropolis<ProjectorProbabilityFunction2>(&markow2,private_samples,(size_t) (iterations/3));
#pragma omp critical
			{
			for(std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: private_samples){
				auto itr = samples.find(pair.first);
				if (itr == samples.end()){
					samples[pair.first].first = pair.second.first;
					samples[pair.first].second = pair.second.second;
				} else
					samples[pair.first].first += pair.second.first;
			}
			}

			}

			XERUS_LOG(info,"Start eHx");

			auto samples_keys = extract_keys(samples,accuracy);
			//Calculate eHx for relevant samples
#pragma omp parallel for schedule(dynamic) shared(eHxValues) firstprivate(builder)
						for (size_t i = 0; i < samples_keys.size(); ++i){
							auto itr = eHxValues.find(samples_keys[i]);
							if (itr == eHxValues.end()){
								 //setting builder to newest sample!! Important
								builder.reset(samples_keys[i]);
							  value_t tmp = builder.contract();
#pragma omp critical
					eHxValues[samples_keys[i]] = tmp;
				}
			}
			XERUS_LOG(info,"Start component computation");
			//Calculate tangential component
			Tensor result(phi.component(pos).dimensions);
			for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: samples) {
				if ((value_t) pair.second.first > accuracy * (value_t) iterations){
					auto loc_grad = uvP.localProduct(pair.first,pos,true);
					auto idx = makeIndex(pair.first);
					builder.reset(pair.first);
					dk = builder.diagionalEntry();
					factor = (eHxValues[pair.first] - ev*phi[idx])/dk;
					result += factor * loc_grad;
				}
			}
			XERUS_LOG(info,"Pos = " << pos << " " << result.frob_norm() << " Number of samples " << samples.size());
			results.emplace_back(result);
		}
	return results;
	}

		std::vector<Tensor> get_2_tangential_components(value_t ev, size_t start_pos, value_t accuracy=0.0001){
			value_t ev_exact,res,psi_ek,factor,dk;
			std::vector<Tensor> results; // Vector of Tangential space components
			xerus::TTOperator Hs,Fock_inv;
		  std::string name2 = "/homes/numerik/goette/Documents/eclipse-workspace/Xerus_Sandbox/QC-Experiments/H2O/data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
			read_from_disc(name2,Hs);
			name2 = "../data/fock_h2o_inv_shifted" + std::to_string(d) +"_full.ttoperator";
			read_from_disc(name2,Fock_inv);

			//for (size_t pos = start_pos; pos < start_pos+1; ++pos){
			for (size_t pos = 0; pos < d; ++pos){
				value_t prob = 0, prob2  = 0;
				ProjectorProbabilityFunction3 PPF(phi,pos, true,Fock_inv);
				Metropolis<ProjectorProbabilityFunction3> markow2(&PPF,  /*TODO check*/ TrialSample, start_sample, d);

				std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples;
				runMetropolis<ProjectorProbabilityFunction3>(&markow2,samples,iterations);

				size_t count = 0;
				Tensor result(phi.component(pos).dimensions);
				for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: samples) {
					prob +=pair.second.second; //TODO check this
					if ((value_t) pair.second.first > accuracy * (value_t) iterations){
						prob2 +=pair.second.second; //TODO check this
						count++;
						auto itr = eHxValues.find(pair.first);
						if (itr == eHxValues.end()){
							 //setting builder to newest sample!! Important
							builder.reset(pair.first);
							eHxValues[pair.first] = builder.contract();
						}

						auto loc_grad = uvP.localProduct(pair.first,pos,true);
						auto idx = makeIndex(pair.first);

						builder.reset(pair.first);
						//dk = builder.diagionalEntry();
						//factor = (eHxValues[pair.first] - ev*phi[idx])/dk;

						auto idx2 = makeIndex(pair.first);
						idx2.resize(2 * d);
						std::copy_n(idx2.begin(), d, idx2.begin() + d);
						value_t dk = Fock_inv[idx2];
						factor = (eHxValues[pair.first] - ev*phi[idx])*dk;

						result += factor * loc_grad;
					}
				}
				XERUS_LOG(info,"Pos = " << pos << " " << result.frob_norm() << " Number of samples " << samples.size() << " Prob " << prob << " Prob2 " << prob2);
				//XERUS_LOG(info,result);
				results.emplace_back(result);
			}
		return results;
	}


	TTTensor builtTTTensor(const std::vector<Tensor>& y){
		time_t begin_time = time (NULL);
		TTTensor Y(uvP.xbase.first.dimensions);

			for (size_t pos=0;pos<d;pos++){
				Tensor ycomp=y[pos];
				auto dims=ycomp.dimensions;
				auto tmpxl=uvP.xbase.second.get_component(pos);
				auto tmpxr=uvP.xbase.first.get_component(pos);
				if(pos==0){
						ycomp.resize_mode(2,dims[2]*2,0);

						tmpxl.resize_mode(2,dims[2]*2,dims[2]);
						ycomp+=tmpxl;
				}else if(pos==d-1){
						ycomp.resize_mode(0,dims[0]*2,dims[0]);

						tmpxr.resize_mode(0,dims[0]*2,0);
						ycomp+=tmpxr;
				}else{
						ycomp.resize_mode(2,dims[2]*2,0);

						tmpxl.resize_mode(2,dims[2]*2,dims[2]);
						ycomp+=tmpxl;
						ycomp.resize_mode(0,dims[0]*2,dims[0]);
						tmpxr.resize_mode(2,dims[2]*2,0);
						tmpxr.resize_mode(0,dims[0]*2,0);
						ycomp+=tmpxr;

				}
				Y.set_component(pos,ycomp);
			}
			return Y;
		}


	private:


		template<class ProbabilityFunction>
		void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations){
			std::vector<size_t> next_sample;
			for (size_t i = 0; i <  (size_t) (iterations/10); ++i)
				next_sample = markow->getNextSample();

			for (size_t i = 0; i < iterations; ++i){
				next_sample = markow->getNextSample();
				auto itr = umap.find(next_sample);
				if (itr == umap.end()){
					umap[next_sample].first = 1;
					umap[next_sample].second = markow->P->P(next_sample);
				} else
					umap[next_sample].first += 1;

			}
		}

		std::vector<std::vector<size_t>>  extract_keys(std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> const & input_map, value_t accuracy) {
		  std::vector<std::vector<size_t>> retval;
		  for (auto const& element : input_map) {
		  	if ((value_t) element.second.first > accuracy * (value_t) iterations)
		  		retval.push_back(element.first);
		  }
		  return retval;
		}


		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			return index;
		}

		TTTensor makeUnitVector(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
			return unit;
		}

};











