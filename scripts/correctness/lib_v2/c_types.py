"""Exact C storage types for the supported Linux LP64 library ABI."""
import re

MLIR_ELEMENT_TYPES = {
    'float': 'f32', 'double': 'f64',
    'signed char': 'i8', 'unsigned char': 'i8',
    'short': 'i16', 'unsigned short': 'i16',
    'int': 'i32', 'unsigned int': 'i32',
    'long': 'i64', 'unsigned long': 'i64',
    'long long': 'i64', 'unsigned long long': 'i64',
}
INTEGER_TYPES = set(MLIR_ELEMENT_TYPES) - {'float', 'double'}
_FIXED = {'int8_t': 'signed char', 'uint8_t': 'unsigned char',
          'int16_t': 'short', 'uint16_t': 'unsigned short',
          'int32_t': 'int', 'uint32_t': 'unsigned int',
          'int64_t': 'long', 'uint64_t': 'unsigned long'}
_SYNONYMS = {'signed': 'int', 'signed int': 'int', 'unsigned': 'unsigned int',
             'short int': 'short', 'signed short': 'short', 'signed short int': 'short',
             'unsigned short int': 'unsigned short', 'long int': 'long',
             'signed long': 'long', 'signed long int': 'long',
             'unsigned long int': 'unsigned long', 'long long int': 'long long',
             'signed long long': 'long long', 'signed long long int': 'long long',
             'unsigned long long int': 'unsigned long long'}


def canonical_type(spelling, preprocessed=''):
    """Resolve scalar typedefs without losing signedness or accepting pointers."""
    aliases = {alias: underlying for underlying, alias in
               re.findall(r'\btypedef\s+([^;{}]+?)\s+(\w+)\s*;', preprocessed)}
    seen = set()
    value = spelling
    while True:
        value = re.sub(r'\b(?:const|restrict|__restrict|__restrict__)\b', ' ', value)
        value = ' '.join(value.split())
        value = _SYNONYMS.get(value, value)
        if value in MLIR_ELEMENT_TYPES:
            return value
        if value in seen:
            raise ValueError('cyclic C typedef: ' + value)
        seen.add(value)
        if value in aliases:
            value = aliases[value]
        elif value in _FIXED:
            return _FIXED[value]
        else:
            raise ValueError('unsupported or ambiguous C scalar type: ' + value)


def parse_parameter(parameter, preprocessed=''):
    found = re.fullmatch(r'\s*(.+?)\s+(\w+)\s*((?:\[[^\]]*\]\s*)*)', parameter)
    if not found:
        raise ValueError('unsupported C parameter: ' + parameter)
    spelling, name, suffix = found.groups()
    element = canonical_type(spelling, preprocessed)
    dimensions = [dimension.strip() for dimension in re.findall(r'\[([^\]]*)\]', suffix)]
    return name, element, dimensions
