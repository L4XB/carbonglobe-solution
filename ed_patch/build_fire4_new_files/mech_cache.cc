// CarbonGlobe addition: cache the per-year mechanism tables (An, Anb, E, Eb, tf_air, tf_soil) of a SiteData
// on disk so that the 15 seed-age windows of one site-scenario, which see identical forcing and CO2, compute
// each (species, month, light-bin) Farquhar entry only once. Exact: with do_downreg = 0 the cohort Vm0 is
// Vm0_max[species], so every entry is a pure function of (species, month, light bin, year). Entries never
// computed stay at -9999 and are filled lazily by compute_mech as before. Enabled by ED_MECH_CACHE_DIR.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "edmodels.h"
#include "read_site_data.h"
#include "mech_cache.h"

static const double MISSING = -1000.0;   // compute_mech's "not computed" test is value < -1000
static int g_last_year = -1;

// with ED_YEAR_CYCLE=n (and ED_WINDOW_YEAR0=y0) the yearly input files cycle over y0 .. y0+n-1 (read_site_data.cc):
// the tables of a simulated year are those of its FILE year
static int cycle_year(int year) {
   const char* c = getenv("ED_YEAR_CYCLE"); const char* y = getenv("ED_WINDOW_YEAR0");
   if (!c || !y || year < 0) return year;
   int n = atoi(c), y0 = atoi(y);
   if (n <= 0) return year;
   int k = (year - y0) % n; if (k < 0) k += n;
   return y0 + k;
}

static std::string cache_path(int year) {
   const char* dir = getenv("ED_MECH_CACHE_DIR");
   if (dir == NULL) return std::string();
   year = cycle_year(year);
   char buf[1024];
   snprintf(buf, sizeof buf, "%s/mech_%d.bin", dir, year);
   return std::string(buf);
}

struct MechBlob {
   double An[NSPECIES][N_CLIMATE][N_LIGHT];
   double Anb[NSPECIES][N_CLIMATE][N_LIGHT];
   double E[NSPECIES][N_CLIMATE][N_LIGHT];
   double Eb[NSPECIES][N_CLIMATE][N_LIGHT];
   double tf_air[NSPECIES][N_CLIMATE];
   double tf_soil[NSPECIES][N_CLIMATE];
};

static void blob_from_site(const SiteData* s, MechBlob* b) {
   memcpy(b->An, s->An, sizeof b->An); memcpy(b->Anb, s->Anb, sizeof b->Anb);
   memcpy(b->E, s->E, sizeof b->E); memcpy(b->Eb, s->Eb, sizeof b->Eb);
   memcpy(b->tf_air, s->tf_air, sizeof b->tf_air); memcpy(b->tf_soil, s->tf_soil, sizeof b->tf_soil);
}

static bool read_blob(const std::string& path, MechBlob* b) {
   FILE* f = fopen(path.c_str(), "rb");
   if (!f) return false;
   size_t n = fread(b, sizeof(MechBlob), 1, f);
   fclose(f);
   return n == 1;
}

// merge: take entries that are computed (>= -1000) in `from` into `into` where `into` is missing
template <size_t N> static void merge_arr(double (&into)[N], const double (&from)[N]) {
   for (size_t i = 0; i < N; i++) if (into[i] < MISSING && from[i] >= MISSING) into[i] = from[i];
}
static void merge_blob(MechBlob* into, const MechBlob* from) {
   merge_arr(*reinterpret_cast<double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(into->An), *reinterpret_cast<const double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(from->An));
   merge_arr(*reinterpret_cast<double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(into->Anb), *reinterpret_cast<const double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(from->Anb));
   merge_arr(*reinterpret_cast<double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(into->E), *reinterpret_cast<const double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(from->E));
   merge_arr(*reinterpret_cast<double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(into->Eb), *reinterpret_cast<const double (*)[NSPECIES * N_CLIMATE * N_LIGHT]>(from->Eb));
   merge_arr(*reinterpret_cast<double (*)[NSPECIES * N_CLIMATE]>(into->tf_air), *reinterpret_cast<const double (*)[NSPECIES * N_CLIMATE]>(from->tf_air));
   merge_arr(*reinterpret_cast<double (*)[NSPECIES * N_CLIMATE]>(into->tf_soil), *reinterpret_cast<const double (*)[NSPECIES * N_CLIMATE]>(from->tf_soil));
}

void mech_cache_save(SiteData* s, int year) {
   std::string path = cache_path(year);
   if (path.empty() || year < 0) return;
   MechBlob* mem = new MechBlob; blob_from_site(s, mem);
   MechBlob* old = new MechBlob;
   if (read_blob(path, old)) merge_blob(mem, old);   // union: memory entries win, file fills the rest
   std::string tmp = path + ".tmp";
   FILE* f = fopen(tmp.c_str(), "wb");
   if (f) { fwrite(mem, sizeof(MechBlob), 1, f); fclose(f); rename(tmp.c_str(), path.c_str()); }
   delete mem; delete old;
}

void mech_cache_load(SiteData* s, int year) {
   g_last_year = year;
   std::string path = cache_path(year);
   if (path.empty()) return;
   MechBlob* b = new MechBlob;
   if (read_blob(path, b)) {
      MechBlob* mem = new MechBlob; blob_from_site(s, mem);
      merge_blob(mem, b);
      memcpy(s->An, mem->An, sizeof s->An); memcpy(s->Anb, mem->Anb, sizeof s->Anb);
      memcpy(s->E, mem->E, sizeof s->E); memcpy(s->Eb, mem->Eb, sizeof s->Eb);
      memcpy(s->tf_air, mem->tf_air, sizeof s->tf_air); memcpy(s->tf_soil, mem->tf_soil, sizeof s->tf_soil);
      delete mem;
   }
   delete b;
}

int mech_cache_last_year(void) { return g_last_year; }
