// Copyright 2024 Jiancong Zhu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg_attr(not(feature = "std"), no_std)]

use bit_vec::BitVec;
use core::{f64::consts::LN_2, marker::PhantomData};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Bloom filter using farmhash.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct Bloom<T: AsRef<[u8]>> {
    /// data of filter in bits.
    bit_vec: BitVec,
    /// number of hash functions.
    k_num: u8,
    /// marker for PhantomData.
    _phantom: PhantomData<T>,
}

impl<T: AsRef<[u8]>> Bloom<T> {
    /// Create a new Bloom filter
    /// `item_count`: number of items expected to be inserted
    /// `false_positive_rate`: expected false positive rate
    pub fn new(item_count: usize, false_positive_rate: f64) -> Self {
        let bits_per_key = Self::bits_per_key(item_count, false_positive_rate);
        let bitmap_size = item_count * bits_per_key;
        let k_num = Self::compute_hash_num(bits_per_key);
        let bit_vec = BitVec::from_elem(bitmap_size, false);
        Self {
            bit_vec,
            k_num,
            _phantom: PhantomData,
        }
    }

    /// Record the presence of an item.
    pub fn set(&mut self, item: &T) {
        let mut h = 0;
        (0..self.k_num).for_each(|i| {
            h = farmhash::hash64_with_seed(item.as_ref(), (i as u64).wrapping_mul(h));
            let bit_pos = (h as usize) % self.bit_vec.len();
            self.bit_vec.set(bit_pos, true);
        })
    }

    /// Check if an item is present in the filter.
    /// There can be false positives, but no false negatives.
    pub fn check(&self, item: &T) -> bool {
        let mut h = 0;
        for i in 0..self.k_num {
            h = farmhash::hash64_with_seed(item.as_ref(), (i as u64).wrapping_mul(h));
            let bit_pos = (h as usize) % self.bit_vec.len();
            if !self.bit_vec.get(bit_pos).unwrap() {
                return false;
            }
        }
        true
    }

    /// Chech and set the item
    /// Return true if the item is already in the filter
    pub fn check_and_set(&mut self, item: &T) -> bool {
        let mut h = 0;
        let mut found = true;
        for i in 0..self.k_num {
            h = farmhash::hash64_with_seed(item.as_ref(), (i as u64).wrapping_mul(h));
            let bit_pos = (h as usize) % self.bit_vec.len();
            if !self.bit_vec.get(bit_pos).unwrap() {
                found = false;
                self.bit_vec.set(bit_pos, true);
            }
        }
        found
    }

    /// Clear the filter, set all bits to 0
    pub fn clear(&mut self) {
        self.bit_vec.clear();
    }

    /// Set all bits to 1, which means the filter is full
    pub fn fill(&mut self) {
        self.bit_vec.set_all();
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        !self.bit_vec.any()
    }

    /// Encode the filter and return a byte vector
    pub fn encode(&self) -> Vec<u8> {
        let mut vec = self.bit_vec.to_bytes();
        vec.push(self.k_num);
        vec
    }

    /// Encode the filter to a byte vector
    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        buf.extend(self.encode());
    }

    /// Decode the filter from a byte vector
    pub fn decode(buf: &[u8]) -> Self {
        let bit_vec = BitVec::from_bytes(&buf[..buf.len() - 1]);
        let k_num = buf[buf.len() - 1];
        Self {
            bit_vec,
            k_num,
            _phantom: PhantomData,
        }
    }

    fn bits_per_key(item_count: usize, fp_rate: f64) -> usize {
        let size = -1.0 * (item_count as f64) * fp_rate.ln() / LN_2.powi(2);
        (size / (item_count as f64)).ceil() as usize
    }

    fn compute_hash_num(bits_per_key: usize) -> u8 {
        assert!(bits_per_key > 0);
        let k = (bits_per_key as f64 * LN_2) as u8;
        k.clamp(1, 25)
    }
}

#[cfg(all(feature = "std", test))]
mod tests {
    use super::*;

    #[test]
    fn test_bloom() {
        let mut bloom = Bloom::new(1000, 0.01);
        for i in 0..100 {
            bloom.set(&i.to_string());
        }
        for i in 0..100 {
            assert!(bloom.check(&i.to_string()));
        }
    }

    fn false_positive_rate_case(item_count: usize, test_count: usize, fp_rate: f64) {
        let mut bloom = Bloom::new(item_count, fp_rate);
        for i in 0..item_count {
            bloom.set(&format!("key{}", i));
        }
        let mut false_positive = 0f64;
        for i in item_count..item_count + test_count {
            if bloom.check(&format!("key{}", i)) {
                false_positive += 1.0;
            }
        }
        let rate = false_positive / test_count as f64;
        assert!(rate < fp_rate * 3.0);
    }

    #[test]
    fn test_bloom_bits_per_key() {
        false_positive_rate_case(1000000, 100000, 0.1);
        false_positive_rate_case(1000000, 100000, 0.01);
        false_positive_rate_case(1000000, 100000, 0.001);
    }
}
