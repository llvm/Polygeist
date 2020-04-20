int main() {
#pragma scop 
  int a = 0;
  for (int k = 0; k < 100; k++)
    a = a + 1;

#pragma endscop
  return a;
}
