#include "otherInterface.h"

long long htonll(long long src)
{
    long long dst(0),temp(0);
    char buf[8];
    memcpy(buf,&src,8);
    int idx(0);
    int offset(56);
    while(idx < 8)
    {
        temp = buf[idx++];
        dst |= (temp << offset);
        offset -= 8;
    }
    return dst;
}
