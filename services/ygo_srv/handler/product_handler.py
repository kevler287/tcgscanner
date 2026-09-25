from typing import List, Optional

from fastapi import Request, Response
from pydantic import BaseModel
from tcgs.yugioh.setcode_resolver import *

class ErrorResponse(BaseModel):
    msg: str
    setcode_input: str
    expansioncode_options: Optional[List[str]] = None
    collectornumber_options: Optional[List[str]] = None
    language_code: Optional[str] = None

def identify_product(request: Request, setcode: str):
    parts = setcode.split("-")
    if len(parts) != 2:
        return Response(content=f"setcode {setcode} has no hyphen", status_code=500)

    expansion_code = parts[0]
    language_code = parts[1][:-3]
    collector_number = parts[1][-3:]

    ec_opts = resolve_expansion_code(expansion_code)
    language_code, language = resolve_language_code(language_code)
    cn_opts = resolve_collector_number(collector_number)

    catalog_df = request.app.state.catalog
    remaining_entries = catalog_df[catalog_df["expansionCode"].isin(ec_opts)]

    if len(remaining_entries) == 0:
        return Response(status_code=404, content=ErrorResponse(
            msg=f"no products remain after expansioncode filtering",
            setcode_input=setcode,
            expansioncode_options=ec_opts
        ))

    remaining_entries = remaining_entries[remaining_entries["collectorNumber"].isin(cn_opts)]
    if len(remaining_entries) == 0:
        return Response(status_code=404, content=ErrorResponse(
            msg=f"no products remain after collectornumber filtering",
            setcode_input=setcode,
            expansioncode_options=ec_opts,
            collectornumber_options=cn_opts
        ))
        
    if len(remaining_entries) == 1:
        product = remaining_entries.iloc[0].to_dict()
        product["x-language"] = language
        return product

    if language_code:
        remaining_entries = remaining_entries[
            remaining_entries["lang_codes"].apply(
                lambda codes: language_code in codes or "*" in codes
            )
        ]
    else:
        remaining_entries = remaining_entries[
            remaining_entries["lang_codes"].apply(lambda codes: codes == [])
        ]

    if len(remaining_entries) == 0:
        return Response(status_code=404, content=ErrorResponse(
            msg=f"no products remain after languagecode filtering",
            setcode_input=setcode,
            expansioncode_options=ec_opts,
            collectornumber_options=cn_opts,
            language_code=language_code
        ))
        
    if len(remaining_entries) == 1:
        product = remaining_entries.iloc[0].to_dict()
        product["x-language"] = language
        return product

    return Response(status_code=409, content=ErrorResponse(
        msg=f"{len(remaining_entries)} products remain after languagecode filtering",
        setcode_input=setcode,
        expansioncode_options=ec_opts,
        collectornumber_options=cn_opts,
        language_code=language_code
    ))
