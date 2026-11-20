#ifndef CGB_MECH_CACHE_H_
#define CGB_MECH_CACHE_H_
struct SiteData;
void mech_cache_save(SiteData* s, int year);
void mech_cache_load(SiteData* s, int year);
int mech_cache_last_year(void);
#endif
