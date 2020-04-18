int main() {
#pragma scop 
  int k;
  k = 23; 
  
  float i = 100.0;

#pragma endscop
  return i;
}
