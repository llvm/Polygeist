
int main() {
#pragma scop
  float a = 0.0;
  for (int i = 0; i < 100; i++)
    for (int j = 3 * i + 5; j < 2000; j++)
      a++;
#pragma endscop
  return a;
}
