static uint64_t seed;

void splitmix64_next()
{
    uint64_t z = (seed += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    seed = z ^ (z >> 31);
}

void devRandomSeed()
{
    int urnd = open("/dev/random", O_RDONLY);
    read(urnd, &seed, 8);
    close(urnd);
}

void seeds(uint64_t *s, int n){
    for( int i = 0; i < 4*n; i++){
        if (i==0) devRandomSeed();
        splitmix64_next(); s[i] = seed;
    }
}
