#[cfg(test)]
mod test;
pub(crate) mod whale;

use crate::problem::{Individual, Solvable};

use crate::jssp::whale::Whale;
use nalgebra::DVector;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, Error, Read};

#[derive(Debug, PartialEq, Clone)]
#[allow(unused)]
pub enum Instance {
    TEST01,
    ABZ05,
    FT06,
    LA02,
    LA05,
    LA07,
}
impl Display for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name: &str = match self {
            Instance::TEST01 => "test01",
            Instance::ABZ05 => "abz05",
            Instance::FT06 => "ft06",
            Instance::LA02 => "la02",
            Instance::LA05 => "la05",
            Instance::LA07 => "la07",
        };
        write!(f, "{name}")
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct JobShopSchedulingProblem {
    instance: Instance,
    sequences: Vec<Vec<usize>>,
    processing_times: Vec<Vec<f64>>,
    pub(crate) n_jobs: usize,
    pub(crate) n_machines: usize,
}

impl JobShopSchedulingProblem {
    fn new(
        instance: Instance,
        sequences: Vec<Vec<usize>>,
        processing_times: Vec<Vec<f64>>,
        n_jobs: usize,
        n_machines: usize,
    ) -> Self {
        Self {
            instance,
            sequences,
            processing_times,
            n_jobs,
            n_machines,
        }
    }

    fn read_instance_file(path: &str) -> Result<String, Error> {
        let file = File::open(path)?;
        let mut content: String = String::new();
        BufReader::new(file).read_to_string(&mut content)?;
        Ok(content)
    }

    fn read_sequences(lines: &Vec<Vec<f64>>) -> Vec<Vec<usize>> {
        let mut sequences: Vec<Vec<usize>> = vec![];
        for l in lines {
            let mut i: usize = 0;
            let mut aux: Vec<usize> = vec![];
            while i < l.len() {
                aux.push(l[i] as usize);
                i += 2;
            }
            sequences.push(aux);
        }
        sequences
    }

    fn read_processing_times(lines: &[Vec<f64>], sequences: &[Vec<usize>]) -> Vec<Vec<f64>> {
        let mut processing_times: Vec<Vec<f64>> = vec![];
        for i in 0..lines.len() {
            let l = &lines[i];
            let sequence = &sequences[i];
            let mut aux: Vec<f64> = vec![0f64; sequence.len()];
            let mut j = 1usize;
            for next_index in sequence {
                aux[*next_index] = l[j];
                j += 2;
            }
            processing_times.push(aux);
        }
        processing_times
    }

    pub fn from_instance(instance: Instance) -> Result<Self, Error> {
        let content: String = Self::read_instance_file(&format!("lit/{instance}.txt"))?;
        let mut lines = content
            .lines()
            .map(|line| {
                line.replace("\t", " ")
                    .split_whitespace()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        let first_line = lines.remove(0);
        let [n_jobs, n_machines] = first_line[..] else {
            panic!("Could not parse first line.")
        };
        let sequences = Self::read_sequences(&lines);
        let processing_times = Self::read_processing_times(&lines, &sequences);

        Ok(Self::new(
            instance,
            sequences,
            processing_times,
            n_jobs as usize,
            n_machines as usize,
        ))
    }

    pub fn generate_base_sequence(&self) -> DVector<usize> {
        let size: usize = self.n_jobs * self.n_machines;
        let mut seq: DVector<usize> = DVector::zeros(size);
        let mut s: usize = 0;
        for i in 1..=size {
            seq[i - 1] = s;
            if i % self.n_machines == 0 {
                s += 1;
            }
        }
        seq
    }
}

impl Solvable<usize, Whale> for JobShopSchedulingProblem {
    #[allow(unused)]
    fn solve(&self, individual: &Whale) -> f64 {
        let mut operations: DVector<usize> = DVector::zeros(self.n_jobs);
        let mut job_time: DVector<f64> = DVector::zeros(self.n_jobs);
        let mut machine_time: DVector<f64> = DVector::zeros(self.n_machines);
        for job in individual.solution_vector() {
            let job: usize = *job;
            // Machine of given operation of the current job
            let machine: usize = self.sequences[job][operations[job]];
            let start = job_time[job].max(machine_time[machine]);
            let finish = start + self.processing_times[job][machine];
            job_time[job] = finish;
            machine_time[machine] = finish;
            operations[job] += 1;
        }
        job_time.max()
    }
}
