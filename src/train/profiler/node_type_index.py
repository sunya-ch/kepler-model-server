# NodeTypeIndexCollection manages node_type index under pipeline path
#p
# Usage:
#   index_collection = NodeTypeIndexCollection(pipeline_path)
#   node_type = index_collection.index_train_machine(machine_id, new_spec)
#   index_collection.save()

import sys
import os

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from saver import save_node_type_index, save_machine_spec
from loader import load_node_type_index
from train_types import NodeAttribute
from similarity_utils import compute_jaccard_similarity, get_similarity_weight, compute_similarity, get_num_of_none, get_candidate_score, find_best_candidate, compute_uncertainty, compute_looseness

import enum

import re
def rename(name):
    name = name.replace("(R)", "")
    name = name.replace("(r)", "")
    name = name.replace("CPU", "")
    name = name.replace("Processor", "")
    name = name.replace("processor", "")
    name = re.sub("\d+-Bit Multi-Core", '', name)
    name = name.split("(")[0].strip()
    name = name.split("[")[0].strip()
    name = name.replace("@", "")
    name = name.replace("®", "")
    name = name.replace(",", "")
    name = re.sub('\d+(\.\d+)?\s?[G|M|g|m][H|h]z', '', name).strip()
    return name

def format_processor(processor):
    return "_".join(re.sub(r'\(.*\)', '', rename(processor)).split()).replace("-", "_").lower().replace("_v", "v")

def format_vendor(vendor):
    return vendor.split()[0].replace("-","_").replace(",","").replace("'","").lower()

GB = 1024*1024*1024
import psutil
import cpuinfo
import subprocess

import pyudev

def discover_spec_values():
    processor = ""
    vendor = ""
    cpu_info = cpuinfo.get_cpu_info()
    if "brand_raw" in cpu_info:
        processor = format_processor(cpu_info["brand_raw"])
    context = pyudev.Context()
    for device in context.list_devices(subsystem="dmi"):
        if device.get('ID_VENDOR') is not None:
            vendor = format_vendor(device.get('ID_VENDOR'))
            break
    if vendor == "" and "vendor_id_raw" in cpu_info:
        vendor = format_vendor(cpu_info["vendor_id_raw"])
        
    cores = psutil.cpu_count(logical=True)
    chips = max(1, int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True)))
    threads_per_core = max(1, cores//psutil.cpu_count(logical=False))
    memory = psutil.virtual_memory().total
    memory_gb = int(memory/GB)
    freq = psutil.cpu_freq(percpu=False)
    spec_values = {
        "vendor": vendor,
        "processor": processor,
        "cores": cores,
        "chips": chips,
        "memory": memory_gb,
        "threads_per_core": threads_per_core
    }
    if freq is not None:
        cpu_freq_mhz = round(max(freq.max, freq.current)/100)*100 # round to one decimal of GHz
        spec_values["frequency"] = cpu_freq_mhz
    return spec_values

def generate_spec(data_path, machine_id):
    spec_values = discover_spec_values()
    spec = NodeTypeSpec(**spec_values)
    print("Save machine spec ({}): ".format(data_path))
    print(str(spec))
    save_machine_spec(data_path, machine_id, spec)

def load_node_type_spec(node_type_index_json):
    node_type_spec_index = dict()
    if node_type_index_json is not None:
        for index, spec_obj in node_type_index_json.items():
            spec = NodeTypeSpec()
            spec.load(spec_obj)
            node_type_spec_index[int(index)] = spec
    return node_type_spec_index

no_data = None

# NodeTypeSpec defines spec of each node_type index
class NodeTypeSpec():
    def __init__(self, **kwargs):
        self.attrs = dict()        
        self.attrs[NodeAttribute.PROCESSOR] = kwargs.get('processor', no_data)
        self.attrs[NodeAttribute.CORES] = kwargs.get('cores', no_data)
        self.attrs[NodeAttribute.CHIPS] = kwargs.get('chips', no_data)
        self.attrs[NodeAttribute.MEMORY] = kwargs.get('memory', no_data)
        self.attrs[NodeAttribute.FREQ] = kwargs.get('frequency', no_data)
        self.members = []

    def is_none(self):
        for value in self.attrs.values():
            if value != no_data:
                return False
        return True
        
    def load(self, json_obj):
        for attr, attr_values in json_obj["attrs"].items():
            self.attrs[attr] = attr_values
        self.members = json_obj["members"]
        
    def add_member(self, machine_id):
        if machine_id in self.members:
            print("member already exists: ", machine_id)
            return True
        self.members += [machine_id]
        return True    

    def get_size(self):
        return len(self.members)
    
    def get_cores(self):
        return self.attrs[NodeAttribute.CORES]
    
    # check the comparing node-type spec is covered by this node-type spec
    def cover(self, compare_spec):
        if not isinstance(compare_spec, NodeTypeSpec):
            return False
        for attr in NodeAttribute:
            if compare_spec.attrs[attr] is not None:
                try:
                    # Attempt to convert values to floats
                    if float(self.attrs[attr]) != float(compare_spec.attrs[attr]):
                        return False
                except ValueError:
                    # If conversion to float fails, compare as strings
                    if self.attrs[attr] != compare_spec.attrs[attr]:
                        return False
        return True

    def __str__(self):
        out_str = ""
        for attr in NodeAttribute:
            out_str += "{} ({})\n".format(attr, str(self.attrs[attr]))
        return out_str
    
    def get_uncertain_attribute_freq(self, compare_spec):
        uncertain_attribute_freq = dict()
        if not self.cover(compare_spec):
            # not covered
            return None
        size = self.get_size()
        for attr in NodeAttribute:
            if compare_spec.attrs[attr] is None:
                uncertain_attribute_freq[attr] = size
        return uncertain_attribute_freq
    
    def get_json(self):
        json_obj = dict()
        json_obj["attrs"] = dict()
        for attr in NodeAttribute:
            json_obj["attrs"]["{}".format(attr)] = self.attrs[attr]
        json_obj["members"] = self.members
        return json_obj
    
    def complete_info(self):
        for attr in NodeAttribute:
            if self.attrs[attr] is None:
                return False
        return True

    def get_similarity(self, compare_spec, debug=False):
        total_similarity = 0
        for attr in NodeAttribute:
            similarity = 0
            # compare similar string
            if compare_spec.attrs[attr] is not None and attr in [NodeAttribute.PROCESSOR]:
                similarity = compute_jaccard_similarity(self.attrs[attr], compare_spec.attrs[attr])
            # compare number
            else:
                if compare_spec.attrs[attr] is not None:
                    similarity = compute_similarity(self.attrs[attr], compare_spec.attrs[attr])
            if debug:
                print(attr, self.attrs[attr], compare_spec.attrs[attr], similarity, get_similarity_weight(attr))
            assert similarity >= 0, "similarity must be >= 0, {} ({})".format(attr, similarity)
            assert similarity <= 1, "similarity must be <= 1, {} ({})".format(attr, similarity)
            total_similarity += (similarity*get_similarity_weight(attr))
        if total_similarity > 1:
            total_similarity = 1
        return total_similarity
    
    def copy(self):
        spec = NodeTypeSpec()
        spec.attrs = self.attrs.copy()
        spec.members = self.members.copy()
        return spec
        
    
class NodeTypeIndexCollection():
    def __init__(self, pipeline_path):
        self.pipeline_path = pipeline_path
        node_type_index_json = load_node_type_index(self.pipeline_path)
        self.node_type_index = load_node_type_spec(node_type_index_json)

    def index_train_machine(self, machine_id, new_spec):
        if not new_spec.complete_info():
            print("Machine info not completed: ", str(new_spec))
            return -1
        covered_index, _, _ = self.get_node_type(new_spec)
        if covered_index == -1:
            covered_index = 0
            if len(self.node_type_index.keys()) > 0:
                covered_index = max(self.node_type_index.keys()) + 1
            self.node_type_index[covered_index] = new_spec
        self.node_type_index[covered_index].add_member(machine_id)
        return covered_index

    def get_node_type(self, in_spec, loose_search=False, debug=False):
        if len(self.node_type_index) == 0:
            return -1, -1, -1
        compare_spec = in_spec.copy()
        num_of_none = get_num_of_none(compare_spec)
        similarity_map, max_similarity, most_similar_index, has_candidate, candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total = self._find_candidates(in_spec, loose_search, debug)
        if max_similarity == 1:
            return most_similar_index, 0, 0
        if has_candidate:
            # covered
            candidate_score = get_candidate_score(candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total)
            best_candidate_index, max_score = find_best_candidate(candidate_score)
            uncertainty = compute_uncertainty(max_score, num_of_none)
            return best_candidate_index, uncertainty, 0
        else:
            # not covered
            if loose_search:
                if most_similar_index != -1:
                    candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total, num_of_none = self._loose_search(compare_spec, debug, similarity_map, max_similarity, most_similar_index, candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total)
                    candidate_score = get_candidate_score(candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total)
                    if debug:
                        print("Candidate score: ", candidate_score)
                    most_similar_score = candidate_score[most_similar_index]
                    uncertainty = compute_uncertainty(most_similar_score, num_of_none)
                    if max_similarity != -1:
                        looseness = compute_looseness(max_similarity)
                        return most_similar_index, uncertainty, looseness
        return -1, -1, -1
    
    
    def get_json(self):
        json_obj = dict()
        for index, node_type_spec in self.node_type_index.items():
            json_obj[index] = node_type_spec.get_json()
        return json_obj
    
    def save(self):
        obj = self.get_json()
        save_node_type_index(self.pipeline_path, obj)

    def copy(self):
        node_collection = NodeTypeIndexCollection(self.pipeline_path)
        removed_items = [node_type for node_type in node_collection.node_type_index.keys() if node_type not in self.node_type_index.keys()]
        for node_type in removed_items:
            del node_collection.node_type_index[node_type]
        return node_collection

    def _find_candidates(self, compare_spec, loose_search=False, debug=False):
        """
        Returns:
        - similarity_map: map from node type index to similarity value
        """
        candidate_uncertain_attribute_freq = dict()
        candidate_uncertain_attribute_total = dict()
        most_similar_index = -1
        max_similarity = -1
        most_similar_freq = -1
        completed_info = compare_spec.complete_info()
        has_candidate = False
        similarity_map = dict()
        for attr in NodeAttribute:
            candidate_uncertain_attribute_freq[attr] = []
            candidate_uncertain_attribute_total[attr] = 0
        for index, node_type_spec in self.node_type_index.items():
            freq = node_type_spec.get_size()
            if loose_search:
                similarity = node_type_spec.get_similarity(compare_spec)
                similarity_map[index] = similarity
                if similarity > max_similarity or (similarity == max_similarity and most_similar_freq < freq):
                    most_similar_index = index
                    max_similarity = similarity
                    most_similar_freq = freq
                    if debug:
                        print(index, str(node_type_spec), similarity)
            if node_type_spec.cover(compare_spec):
                if completed_info:
                    return similarity_map, 1, index, has_candidate, candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total
                else:
                    for attr in NodeAttribute:
                        if compare_spec.attrs[attr] is None: 
                            candidate_uncertain_attribute_freq[attr] += [(index, freq)]
                            candidate_uncertain_attribute_total[attr] += freq
                            has_candidate = True
        return similarity_map, max_similarity, most_similar_index, has_candidate, candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total

    def _loose_search(self, compare_spec, debug, similarity_map, max_similarity, most_similar_index, candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total):
        num_of_none = get_num_of_none(compare_spec)
        most_similar_spec = self.node_type_index[most_similar_index]
        # remove uncovered spec
        for attr in NodeAttribute:
            if compare_spec.attrs[attr] != most_similar_spec.attrs[attr]:
                if debug:
                    print("Loosen {} ({}-->{})".format(attr, compare_spec.attrs[attr], most_similar_spec.attrs[attr]))
                compare_spec.attrs[attr] = None
                num_of_none += 1
        # find uncertainty
        for index, node_type_spec in self.node_type_index.items():
            if node_type_spec.cover(compare_spec):
                similarity = similarity_map[index]
                freq = node_type_spec.get_size()
                if similarity == max_similarity and freq > self.node_type_index[most_similar_index].get_size():
                    if debug:
                        print("change most similar index from {} to {}".format(most_similar_index, index))
                    most_similar_index = index
                for attr in NodeAttribute:
                    if compare_spec.attrs[attr] is None: 
                        candidate_uncertain_attribute_freq[attr] += [(index, freq)]
                        candidate_uncertain_attribute_total[attr] += freq
            if most_similar_index == index:
                node_type_spec.cover(compare_spec)
        return candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total, num_of_none
                    
